#!/usr/bin/env python3
"""
diarize.py - Speaker diarization using ONNX models (CPU only).

Pipeline: pyannote segmentation → wespeaker embedding → spectral clustering.
Runs entirely on CPU, keeping the NPU free for Whisper transcription.
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

SAMPLE_RATE = 16000
SEG_WINDOW_S = 10           # segmentation model input window
SEG_HOP_S = 2               # sliding window hop
SEG_SAMPLES = SEG_WINDOW_S * SAMPLE_RATE  # 160,000
NUM_POWERSET_CLASSES = 7    # pyannote 3.0 powerset output classes
EMBEDDING_DIM = 192
MIN_SEGMENT_S = 1.5         # merge segments shorter than this
BINARIZE_THRESHOLD = 0.5

# Powerset class mapping for pyannote-segmentation-3.0
# 7 classes encode speaker combinations (up to 3 speakers, max 2 simultaneous):
#   0: no speech
#   1: speaker 1 only
#   2: speaker 2 only
#   3: speaker 3 only
#   4: speakers 1+2
#   5: speakers 1+3
#   6: speakers 2+3
POWERSET_MAPPING = {
    0: [],
    1: [0],
    2: [1],
    3: [2],
    4: [0, 1],
    5: [0, 2],
    6: [1, 2],
}
MAX_SPEAKERS_LOCAL = 3  # max speakers per 10s window


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_segmentation_model(models_dir=None):
    """Load pyannote segmentation ONNX model on CPU."""
    import onnxruntime as ort

    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    model_path = models_dir / "pyannote_segmentation.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Segmentation model not found: {model_path}\n"
            "Run: python download_models.py"
        )

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        str(model_path), sess_options=opts,
        providers=["CPUExecutionProvider"]
    )
    return sess


def load_embedding_model(models_dir=None):
    """Load wespeaker ECAPA-TDNN embedding ONNX model on CPU."""
    import onnxruntime as ort

    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    model_path = models_dir / "wespeaker_ecapa_tdnn.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {model_path}\n"
            "Run: python download_models.py"
        )

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        str(model_path), sess_options=opts,
        providers=["CPUExecutionProvider"]
    )
    return sess


def load_models(models_dir=None):
    """Load both diarization models. Returns (seg_sess, emb_sess)."""
    seg = load_segmentation_model(models_dir)
    emb = load_embedding_model(models_dir)
    return seg, emb


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _decode_powerset(logits):
    """Decode powerset logits to per-frame binary speaker activity.

    Args:
        logits: (num_frames, 7) float32 — raw logits from segmentation model.

    Returns:
        (num_frames, 3) float32 — probability of each speaker being active.
    """
    from scipy.special import softmax
    probs = softmax(logits, axis=-1)  # (num_frames, 7)

    # Accumulate per-speaker probabilities from powerset classes
    activity = np.zeros((logits.shape[0], MAX_SPEAKERS_LOCAL), dtype=np.float32)
    for cls_id, speakers in POWERSET_MAPPING.items():
        for spk in speakers:
            activity[:, spk] += probs[:, cls_id]

    return activity


def run_segmentation(seg_sess, audio, step_callback=None):
    """Run sliding-window segmentation over audio.

    Args:
        seg_sess: ONNX InferenceSession for segmentation model.
        audio: 1-D float32 array, 16 kHz mono.
        step_callback: optional callable(info_dict) for progress reporting.

    Returns:
        list of (start_s, end_s, local_speaker_id) tuples.
    """
    total_samples = len(audio)
    total_duration = total_samples / SAMPLE_RATE
    hop_samples = SEG_HOP_S * SAMPLE_RATE

    # Determine window positions
    if total_samples <= SEG_SAMPLES:
        starts = [0]
    else:
        starts = list(range(0, total_samples - SEG_SAMPLES + 1, hop_samples))
        # Ensure we cover the tail
        if starts[-1] + SEG_SAMPLES < total_samples:
            starts.append(total_samples - SEG_SAMPLES)

    # Run first window to discover actual frame count from model output
    # (pyannote-segmentation-3.0 produces 589 frames per 10s window)
    first_window = audio[:SEG_SAMPLES]
    if len(first_window) < SEG_SAMPLES:
        first_window = np.pad(first_window, (0, SEG_SAMPLES - len(first_window)))
    probe_inp = first_window.reshape(1, 1, -1).astype(np.float32)
    probe_out = seg_sess.run(None, {"input_values": probe_inp})
    frames_per_window = probe_out[0].shape[1]
    frame_duration = SEG_WINDOW_S / frames_per_window

    # Total frames for the full audio (at the segmentation resolution)
    total_frames = int(np.ceil(total_duration / frame_duration))
    # Accumulator for stitching overlapping windows
    activity_acc = np.zeros((total_frames, MAX_SPEAKERS_LOCAL), dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)

    for win_idx, start_sample in enumerate(starts):
        # Extract window, pad if needed
        end_sample = start_sample + SEG_SAMPLES
        window = audio[start_sample:end_sample]
        if len(window) < SEG_SAMPLES:
            window = np.pad(window, (0, SEG_SAMPLES - len(window)))

        inp = window.reshape(1, 1, -1).astype(np.float32)

        # Reuse the probe result for the first window
        if win_idx == 0:
            outputs = probe_out
            run_ms = 0.0
        else:
            t0 = time.time()
            outputs = seg_sess.run(None, {"input_values": inp})
            run_ms = (time.time() - t0) * 1000

        if step_callback is not None:
            step_callback({
                "phase": "segmentation",
                "run_time_ms": run_ms,
                "window": win_idx + 1,
                "total_windows": len(starts),
            })

        logits = outputs[0][0]  # (frames_per_window, 7)
        window_activity = _decode_powerset(logits)  # (frames_per_window, 3)

        # Map window frames to global frame indices
        start_time = start_sample / SAMPLE_RATE
        frame_offset = int(round(start_time / frame_duration))

        n_frames = window_activity.shape[0]
        for f in range(n_frames):
            gf = frame_offset + f
            if 0 <= gf < total_frames:
                activity_acc[gf] += window_activity[f]
                counts[gf] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    activity = (activity_acc.T / counts).T.astype(np.float32)  # (total_frames, 3)

    # Binarize
    binary = (activity > BINARIZE_THRESHOLD).astype(np.int32)

    # Convert frame-level binary labels to contiguous segments
    segments = _frames_to_segments(binary, frame_duration, total_duration)

    # Merge short segments
    segments = _merge_short_segments(segments, MIN_SEGMENT_S)

    return segments


def _frames_to_segments(binary, frame_duration, total_duration):
    """Convert (num_frames, 3) binary speaker activity to segment list.

    Returns:
        list of (start_s, end_s, local_speaker_id) sorted by start time.
    """
    num_frames, num_speakers = binary.shape
    segments = []

    for spk in range(num_speakers):
        active = binary[:, spk]
        in_segment = False
        seg_start = 0

        for f in range(num_frames):
            if active[f] and not in_segment:
                seg_start = f
                in_segment = True
            elif not active[f] and in_segment:
                start_s = seg_start * frame_duration
                end_s = f * frame_duration
                if end_s - start_s >= 0.1:  # skip tiny blips
                    segments.append((start_s, end_s, spk))
                in_segment = False

        # Close any open segment
        if in_segment:
            start_s = seg_start * frame_duration
            end_s = min(num_frames * frame_duration, total_duration)
            if end_s - start_s >= 0.1:
                segments.append((start_s, end_s, spk))

    # Sort by start time
    segments.sort(key=lambda s: s[0])
    return segments


def _merge_short_segments(segments, min_duration):
    """Merge segments shorter than min_duration with their nearest neighbor."""
    if len(segments) <= 1:
        return segments

    merged = []
    for seg in segments:
        start, end, spk = seg
        duration = end - start

        if duration < min_duration and merged:
            # Merge with previous segment if same speaker
            prev_start, prev_end, prev_spk = merged[-1]
            if prev_spk == spk:
                merged[-1] = (prev_start, end, spk)
                continue

        merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# Fbank feature extraction (for wespeaker embedding model)
# ---------------------------------------------------------------------------

FBANK_NUM_MELS = 80
FBANK_FRAME_LEN = 0.025    # 25 ms
FBANK_FRAME_SHIFT = 0.010  # 10 ms


def _compute_fbank(audio, sr=SAMPLE_RATE, num_mels=FBANK_NUM_MELS):
    """Compute 80-dim log Fbank features from raw audio.

    Args:
        audio: 1-D float32 array at sr Hz.
        sr: sample rate.
        num_mels: number of mel bins (80 for wespeaker).

    Returns:
        (num_frames, 80) float32 array.
    """
    frame_len = int(sr * FBANK_FRAME_LEN)   # 400 samples
    frame_shift = int(sr * FBANK_FRAME_SHIFT)  # 160 samples
    n_fft = frame_len

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    num_frames = 1 + (len(audio) - frame_len) // frame_shift
    if num_frames < 1:
        # Pad short audio
        audio = np.pad(audio, (0, frame_len - len(audio)))
        num_frames = 1

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_len),
        strides=(audio.strides[0] * frame_shift, audio.strides[0]),
    ).copy()

    # Apply Hamming window
    window = np.hamming(frame_len).astype(np.float32)
    frames *= window

    # FFT → power spectrum
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2 / n_fft

    # Mel filterbank
    low_freq = 20.0
    high_freq = sr / 2.0
    mel_low = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + high_freq / 700.0)
    mel_points = np.linspace(mel_low, mel_high, num_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank_matrix = np.zeros((num_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(num_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fbank_matrix[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fbank_matrix[m, k] = (f_right - k) / (f_right - f_center)

    mel_spec = power @ fbank_matrix.T  # (num_frames, num_mels)
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    # CMN (cepstral mean normalization)
    mel_spec -= mel_spec.mean(axis=0, keepdims=True)

    return mel_spec.astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(emb_sess, audio, segments, step_callback=None):
    """Extract speaker embeddings for each segment.

    Args:
        emb_sess: ONNX InferenceSession for embedding model.
        audio: 1-D float32 array, 16 kHz mono.
        segments: list of (start_s, end_s, local_speaker_id).
        step_callback: optional callable(info_dict) for progress.

    Returns:
        (N, 192) float32 array of speaker embeddings.
    """
    embeddings = []
    total = len(segments)

    for idx, (start_s, end_s, _spk) in enumerate(segments):
        start_sample = int(start_s * SAMPLE_RATE)
        end_sample = int(end_s * SAMPLE_RATE)
        segment_audio = audio[start_sample:end_sample]

        # Skip very short segments
        if len(segment_audio) < SAMPLE_RATE // 4:  # < 0.25s
            embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
            continue

        # Extract Fbank features (80-dim) for wespeaker model
        feats = _compute_fbank(segment_audio)  # (T, 80)
        inp = feats.reshape(1, -1, FBANK_NUM_MELS).astype(np.float32)

        t0 = time.time()
        outputs = emb_sess.run(None, {"feats": inp})
        run_ms = (time.time() - t0) * 1000

        if step_callback is not None:
            step_callback({
                "phase": "embedding",
                "run_time_ms": run_ms,
                "segment": idx + 1,
                "total_segments": total,
            })

        emb = outputs[0].flatten()[:EMBEDDING_DIM]
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_speakers(embeddings):
    """Cluster speaker embeddings using spectral clustering.

    Args:
        embeddings: (N, 192) float32 array.

    Returns:
        list of int — cluster labels (0, 1, 2, ...).
    """
    from spectralcluster import SpectralClusterer

    if len(embeddings) <= 1:
        return [0] * len(embeddings)

    # Replace any NaN/zero embeddings with small random noise to avoid
    # division-by-zero in spectralcluster's normalization
    clean = embeddings.copy()
    for i in range(len(clean)):
        norm = np.linalg.norm(clean[i])
        if norm < 1e-8 or np.any(np.isnan(clean[i])):
            clean[i] = np.random.randn(clean.shape[1]).astype(np.float32) * 1e-4
            clean[i] /= np.linalg.norm(clean[i])

    clusterer = SpectralClusterer(
        min_clusters=1,
        max_clusters=10,
        refinement_options=None,
    )

    labels = clusterer.predict(clean)
    return labels.tolist()


# ---------------------------------------------------------------------------
# Top-level diarization
# ---------------------------------------------------------------------------

def diarize(seg_sess, emb_sess, audio, step_callback=None):
    """Run full diarization pipeline.

    Args:
        seg_sess: segmentation ONNX session.
        emb_sess: embedding ONNX session.
        audio: 1-D float32 array, 16 kHz mono.
        step_callback: optional callable(info_dict) for CPU activity reporting.

    Returns:
        list of (start_s, end_s, speaker_id) sorted by time.
        speaker_id is a globally consistent label (0, 1, 2, ...).
    """
    # Step 1: Segmentation
    segments = run_segmentation(seg_sess, audio, step_callback=step_callback)

    if not segments:
        # No speech detected — return single segment for the whole audio
        duration = len(audio) / SAMPLE_RATE
        return [(0.0, duration, 0)]

    # Step 2: Embedding extraction
    embeddings = extract_embeddings(
        emb_sess, audio, segments, step_callback=step_callback
    )

    # Step 3: Spectral clustering
    labels = cluster_speakers(embeddings)

    # Map local segment speaker IDs to global cluster labels
    result = []
    for (start_s, end_s, _local_spk), global_spk in zip(segments, labels):
        result.append((start_s, end_s, global_spk))

    # Sort by time and merge consecutive same-speaker segments
    result.sort(key=lambda s: s[0])
    result = _merge_consecutive(result)

    return result


def _merge_consecutive(segments, gap_threshold=2.0):
    """Merge consecutive segments with the same speaker ID.

    If two adjacent segments have the same speaker and the gap between
    them is less than gap_threshold seconds, merge into one segment.
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for start, end, spk in segments[1:]:
        prev_start, prev_end, prev_spk = merged[-1]
        if spk == prev_spk and (start - prev_end) < gap_threshold:
            merged[-1] = (prev_start, end, spk)
        else:
            merged.append((start, end, spk))

    return merged


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization using ONNX models (CPU)"
    )
    parser.add_argument("files", nargs="+", help="Audio file(s) to diarize")
    parser.add_argument("--models-dir", default=None,
                        help=f"ONNX models directory (default: {MODELS_DIR})")
    args = parser.parse_args()

    import transcribe_npu

    models_dir = args.models_dir or MODELS_DIR
    print("Loading diarization models (CPU)...")
    t0 = time.time()
    seg_sess, emb_sess = load_models(models_dir)
    print(f"  Models loaded in {time.time() - t0:.1f}s")

    for fpath in args.files:
        print(f"\nDiarizing: {fpath}")

        # Load audio
        audio, sr = transcribe_npu.load_audio(fpath)
        audio = transcribe_npu.resample(audio, sr, SAMPLE_RATE)

        t1 = time.time()
        segments = diarize(seg_sess, emb_sess, audio)
        elapsed = time.time() - t1

        print(f"  Found {len(set(s[2] for s in segments))} speaker(s) "
              f"in {len(segments)} segment(s) ({elapsed:.1f}s)")
        print()
        for start, end, spk in segments:
            m1, s1 = divmod(int(start), 60)
            m2, s2 = divmod(int(end), 60)
            print(f"  [{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] Speaker {spk + 1}")


if __name__ == "__main__":
    main()
