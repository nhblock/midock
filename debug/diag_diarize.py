"""Diagnostic: inspect diarization internals for a specific file."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np
import diarize
import transcribe_npu

FILE = sys.argv[1] if len(sys.argv) > 1 else "test_download2.mp3"
# Region of interest (seconds)
ROI_START = float(sys.argv[2]) if len(sys.argv) > 2 else 280.0
ROI_END = float(sys.argv[3]) if len(sys.argv) > 3 else 320.0

print(f"File: {FILE}")
print(f"ROI: {ROI_START:.0f}s - {ROI_END:.0f}s")

# Load
print("\nLoading models...")
seg_sess, emb_sess = diarize.load_models()

print("Loading audio...")
audio = transcribe_npu.load_audio_16k(FILE)
total_dur = len(audio) / 16000
print(f"  {total_dur:.1f}s of audio")

# ---- Step 1: Raw segmentation (frame-level activity) ----
print("\n=== RAW SEGMENTATION ===")

from scipy.special import softmax

hop_samples = diarize.SEG_HOP_S * diarize.SAMPLE_RATE
total_samples = len(audio)

if total_samples <= diarize.SEG_SAMPLES:
    starts = [0]
else:
    starts = list(range(0, total_samples - diarize.SEG_SAMPLES + 1, hop_samples))
    if starts[-1] + diarize.SEG_SAMPLES < total_samples:
        starts.append(total_samples - diarize.SEG_SAMPLES)

# Probe frame count
first_win = audio[:diarize.SEG_SAMPLES]
if len(first_win) < diarize.SEG_SAMPLES:
    first_win = np.pad(first_win, (0, diarize.SEG_SAMPLES - len(first_win)))
probe = seg_sess.run(None, {"input_values": first_win.reshape(1, 1, -1).astype(np.float32)})
frames_per_window = probe[0].shape[1]
frame_duration = diarize.SEG_WINDOW_S / frames_per_window
print(f"Frames per window: {frames_per_window}, frame_duration: {frame_duration*1000:.1f}ms")

total_frames = int(np.ceil(total_dur / frame_duration))
activity_acc = np.zeros((total_frames, diarize.MAX_SPEAKERS_LOCAL), dtype=np.float64)
counts = np.zeros(total_frames, dtype=np.float64)

for win_idx, start_sample in enumerate(starts):
    window = audio[start_sample:start_sample + diarize.SEG_SAMPLES]
    if len(window) < diarize.SEG_SAMPLES:
        window = np.pad(window, (0, diarize.SEG_SAMPLES - len(window)))
    inp = window.reshape(1, 1, -1).astype(np.float32)

    if win_idx == 0:
        outputs = probe
    else:
        outputs = seg_sess.run(None, {"input_values": inp})

    logits = outputs[0][0]
    window_activity = diarize._decode_powerset(logits)

    start_time = start_sample / diarize.SAMPLE_RATE
    frame_offset = int(round(start_time / frame_duration))

    for f in range(window_activity.shape[0]):
        gf = frame_offset + f
        if 0 <= gf < total_frames:
            activity_acc[gf] += window_activity[f]
            counts[gf] += 1.0

counts = np.maximum(counts, 1.0)
activity = (activity_acc.T / counts).T.astype(np.float32)

# Show frame-level activity in ROI
print(f"\nFrame-level speaker activity in ROI ({ROI_START:.0f}s-{ROI_END:.0f}s):")
print(f"{'Time':>8s}  {'Spk0':>6s}  {'Spk1':>6s}  {'Spk2':>6s}  Active")
roi_start_frame = int(ROI_START / frame_duration)
roi_end_frame = min(int(ROI_END / frame_duration), total_frames)

# Sample every ~1 second
step = max(1, int(1.0 / frame_duration))
for f in range(roi_start_frame, roi_end_frame, step):
    t = f * frame_duration
    a = activity[f]
    active = [i for i in range(3) if a[i] > 0.5]
    active_str = ",".join(f"S{i}" for i in active) if active else "none"
    m, s = divmod(int(t), 60)
    print(f"  {m:02d}:{s:02d}    {a[0]:.3f}   {a[1]:.3f}   {a[2]:.3f}   {active_str}")

# Also show at higher resolution around the speaker change point
print(f"\nHigh-res around {ROI_START + (ROI_END-ROI_START)/2:.0f}s (every ~0.25s):")
mid = (ROI_START + ROI_END) / 2
mid_start = int((mid - 5) / frame_duration)
mid_end = min(int((mid + 5) / frame_duration), total_frames)
step_fine = max(1, int(0.25 / frame_duration))
for f in range(mid_start, mid_end, step_fine):
    t = f * frame_duration
    a = activity[f]
    active = [i for i in range(3) if a[i] > 0.5]
    active_str = ",".join(f"S{i}" for i in active) if active else "none"
    m, s = divmod(int(t), 60)
    frac = t - int(t)
    print(f"  {m:02d}:{s:02d}.{int(frac*10)}  {a[0]:.3f}   {a[1]:.3f}   {a[2]:.3f}   {active_str}")

# ---- Step 2: Full diarization output ----
print("\n=== FULL DIARIZATION (segments in ROI) ===")
segments = diarize.diarize(seg_sess, emb_sess, audio)
print(f"Total: {len(segments)} segments, {len(set(s[2] for s in segments))} speakers")
print(f"\nSegments in ROI:")
for start, end, spk in segments:
    if end >= ROI_START and start <= ROI_END:
        m1, s1 = divmod(int(start), 60)
        m2, s2 = divmod(int(end), 60)
        print(f"  [{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] Speaker {spk + 1}")

# ---- Step 3: Pre-clustering segment details ----
print("\n=== PRE-CLUSTERING SEGMENTS (before embedding) ===")
raw_segments = diarize.run_segmentation(seg_sess, audio)
print(f"Raw segments in ROI (local speaker IDs):")
for start, end, local_spk in raw_segments:
    if end >= ROI_START and start <= ROI_END:
        m1, s1 = divmod(int(start), 60)
        m2, s2 = divmod(int(end), 60)
        print(f"  [{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] local_spk={local_spk}")

# ---- Step 4: Embedding similarity ----
print("\n=== EMBEDDING ANALYSIS ===")
embeddings = diarize.extract_embeddings(emb_sess, audio, raw_segments)
# Find segments near ROI
roi_indices = [i for i, (s, e, _) in enumerate(raw_segments) if e >= ROI_START - 30 and s <= ROI_END + 30]
if len(roi_indices) >= 2:
    print(f"Cosine similarities between nearby segments:")
    for i in roi_indices:
        for j in roi_indices:
            if j > i:
                sim = np.dot(embeddings[i], embeddings[j])
                s1, e1, spk1 = raw_segments[i]
                s2, e2, spk2 = raw_segments[j]
                m1, sec1 = divmod(int(s1), 60)
                m2, sec2 = divmod(int(s2), 60)
                print(f"  [{m1:02d}:{sec1:02d} spk{spk1}] vs [{m2:02d}:{sec2:02d} spk{spk2}]: {sim:.3f}")
