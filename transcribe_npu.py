#!/usr/bin/env python3
"""
transcribe_npu.py - Whisper Large V3 Turbo NPU inference engine

Standalone module using QNN context binaries on Qualcomm Hexagon NPU.
No transformers dependency — implements mel spectrogram, decode loop, and
tokenizer using only onnxruntime-qnn, numpy, scipy, audio2numpy, tokenizers.
"""

import argparse
import os
import time
from math import gcd
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

# Model architecture (Whisper Large V3 Turbo)
NUM_DECODER_LAYERS = 4
D_MODEL = 1280
NUM_HEADS = 20
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
AUDIO_EMB_LEN = 1500
MELS_AUDIO_LEN = 3000
NUM_MEL_BINS = 128
MEAN_DECODE_LEN = 200
VOCAB_SIZE = 51866

# Audio processing
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH_S = 30
N_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE  # 480000

# Decoder tokens
SOT_TOKEN = 50258       # <|startoftranscript|>
EOT_TOKEN = 50257       # <|endoftext|>
LANG_TOKEN = 50259      # <|en|>
TASK_TOKEN = 50360      # <|transcribe|>
NO_TIMESTAMPS = 50364   # <|notimestamps|>
MASK_NEG = np.float16(-100.0)

# Forced decoder sequence
FORCED_TOKENS = [SOT_TOKEN, LANG_TOKEN, TASK_TOKEN, NO_TIMESTAMPS]
NUM_FORCED = len(FORCED_TOKENS)

# Suppress at first generation step (matches Whisper's begin_suppress_tokens)
BEGIN_SUPPRESS_TOKENS = [220, EOT_TOKEN]


# ---------------------------------------------------------------------------
# Mel filterbank (Slaney scale, matching librosa / WhisperFeatureExtractor)
# ---------------------------------------------------------------------------

def _hz_to_mel(freq):
    """Convert Hz to Slaney mel scale."""
    if freq < 1000.0:
        return freq * 3.0 / 200.0
    return 15.0 + 27.0 * np.log(freq / 1000.0) / np.log(6.4)


def _mel_to_hz(mel):
    """Convert Slaney mel scale to Hz."""
    if mel < 15.0:
        return mel * 200.0 / 3.0
    return 1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 27.0)


def _make_mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MEL_BINS,
                         fmin=0.0, fmax=8000.0):
    """
    Create Slaney-normalized mel filterbank.
    Returns (n_mels, n_fft // 2) matching HF WhisperFeatureExtractor
    (Nyquist bin excluded).
    """
    n_freqs = n_fft // 2  # 200 — Nyquist dropped to match HF

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = np.array([_mel_to_hz(m) for m in mels])

    fft_freqs = np.linspace(0, sr / 2, n_freqs + 1)[:n_freqs]  # exclude Nyquist

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        f_left, f_center, f_right = freqs[i], freqs[i + 1], freqs[i + 2]
        if f_center > f_left:
            rising = (fft_freqs - f_left) / (f_center - f_left)
        else:
            rising = np.zeros(n_freqs)
        if f_right > f_center:
            falling = (f_right - fft_freqs) / (f_right - f_center)
        else:
            falling = np.zeros(n_freqs)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
        width = f_right - f_left
        if width > 0:
            fb[i] *= 2.0 / width  # Slaney norm

    return fb


_FILTERBANK = None


def _get_filterbank():
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = _make_mel_filterbank()
    return _FILTERBANK


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

def mel_spectrogram(audio):
    """
    Log-mel spectrogram matching WhisperFeatureExtractor.

    Args:
        audio: 1-D float array, 16 kHz, mono, up to 30 s.

    Returns:
        (1, 128, 3000) float16 array.
    """
    # Pad / truncate to exactly 30 s
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    audio = audio.astype(np.float64)

    # Reflect-free padding for STFT (matches HF: zero-pad n_fft//2 each side)
    audio = np.pad(audio, (N_FFT // 2, N_FFT // 2))

    # Periodic Hann window (matches torch.hann_window periodic=True)
    window = np.hanning(N_FFT + 1)[:-1]

    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    )

    spectrum = np.fft.rfft(frames * window, n=N_FFT)       # (n_frames, 201)
    magnitudes = np.abs(spectrum[:, :-1]) ** 2              # (n_frames, 200)

    fb = _get_filterbank()                                  # (128, 200)
    mel = fb @ magnitudes.T                                 # (128, n_frames)

    # Log scaling
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0

    # Pad / truncate to 3000 frames
    if mel.shape[1] < MELS_AUDIO_LEN:
        mel = np.pad(mel, ((0, 0), (0, MELS_AUDIO_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MELS_AUDIO_LEN]

    return mel.reshape(1, NUM_MEL_BINS, MELS_AUDIO_LEN).astype(np.float16)


# ---------------------------------------------------------------------------
# Audio loading & chunking
# ---------------------------------------------------------------------------

def load_audio(audio_path):
    """Load audio file via audio2numpy (ffmpeg backend). Returns (samples, sr)."""
    from audio2numpy import open_audio
    audio, sr = open_audio(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def resample(audio, orig_sr, target_sr=SAMPLE_RATE):
    """Resample to target_sr using scipy polyphase filter."""
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def chunk_audio(audio, chunk_samples=N_SAMPLES):
    """Split audio into <=30 s chunks."""
    return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]


# ---------------------------------------------------------------------------
# ONNX session loading (QNN Execution Provider)
# ---------------------------------------------------------------------------

def load_sessions(models_dir=None):
    """
    Load encoder + decoder ONNX sessions on the Hexagon NPU.
    Returns (encoder_session, decoder_session).
    """
    import onnxruntime as ort

    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    encoder_onnx = models_dir / "encoder_wrapper.onnx"
    decoder_onnx = models_dir / "decoder_wrapper.onnx"

    for p in (encoder_onnx, decoder_onnx):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    providers = [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})]
    opts = ort.SessionOptions()

    encoder = ort.InferenceSession(str(encoder_onnx), sess_options=opts,
                                   providers=providers)
    decoder = ort.InferenceSession(str(decoder_onnx), sess_options=opts,
                                   providers=providers)
    return encoder, decoder


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the Whisper tokenizer via the ``tokenizers`` library."""
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("openai/whisper-large-v3-turbo")


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------

def transcribe_chunk(encoder_sess, decoder_sess, tokenizer, mel,
                     return_timing=False):
    """
    Transcribe one <=30 s mel spectrogram chunk.

    Args:
        encoder_sess / decoder_sess: ONNX sessions.
        tokenizer: ``tokenizers.Tokenizer``.
        mel: (1, 128, 3000) float16.
        return_timing: if True, return (tokens, timing_dict) instead of tokens.

    Returns:
        list[int] — generated token IDs (special tokens filtered out).
        Or (list[int], dict) when return_timing=True.
    """
    # --- Encoder ---------------------------------------------------------
    t_enc_start = time.time()
    enc_out = encoder_sess.run(None, {"input_features": mel})
    t_enc_end = time.time()
    enc_names = [o.name for o in encoder_sess.get_outputs()]
    cross = dict(zip(enc_names, enc_out))

    # --- Decoder init ----------------------------------------------------
    k_self = [np.zeros((NUM_HEADS, 1, HEAD_DIM, MEAN_DECODE_LEN - 1),
                       dtype=np.float16) for _ in range(NUM_DECODER_LAYERS)]
    v_self = [np.zeros((NUM_HEADS, 1, MEAN_DECODE_LEN - 1, HEAD_DIM),
                       dtype=np.float16) for _ in range(NUM_DECODER_LAYERS)]
    mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, dtype=np.float16)

    dec_out_names = [o.name for o in decoder_sess.get_outputs()]
    generated = []

    t_dec_start = time.time()
    for step in range(MEAN_DECODE_LEN):
        # Input token
        if step < NUM_FORCED:
            tok = FORCED_TOKENS[step]
        else:
            tok = generated[-1]

        # Unmask current position (right → left)
        mask[0, 0, 0, MEAN_DECODE_LEN - 1 - step] = np.float16(0.0)

        feed = {
            "input_ids":      np.array([[tok]], dtype=np.int32),
            "attention_mask":  mask.copy(),
            "position_ids":   np.array([step], dtype=np.int32),
        }
        for i in range(NUM_DECODER_LAYERS):
            feed[f"k_cache_self_{i}_in"] = k_self[i]
            feed[f"v_cache_self_{i}_in"] = v_self[i]
            feed[f"k_cache_cross_{i}"]   = cross[f"k_cache_cross_{i}"]
            feed[f"v_cache_cross_{i}"]   = cross[f"v_cache_cross_{i}"]

        outs = decoder_sess.run(None, feed)
        out = dict(zip(dec_out_names, outs))

        # Update self KV caches
        for i in range(NUM_DECODER_LAYERS):
            k_self[i] = out[f"k_cache_self_{i}_out"]
            v_self[i] = out[f"v_cache_self_{i}_out"]

        # Logits → next token (only after last forced token is fed)
        if step >= NUM_FORCED - 1:
            logits = out["logits"].reshape(-1).astype(np.float32)
            if step == NUM_FORCED - 1:
                for t in BEGIN_SUPPRESS_TOKENS:
                    logits[t] = -np.inf
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            if next_tok == EOT_TOKEN:
                break

    t_dec_end = time.time()

    tokens = [t for t in generated if t < EOT_TOKEN]
    if return_timing:
        timing = {
            "encoder_time_s": t_enc_end - t_enc_start,
            "decoder_time_s": t_dec_end - t_dec_start,
        }
        return tokens, timing
    return tokens


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def transcribe(sessions, tokenizer, audio_path, chunk_callback=None):
    """
    Transcribe an audio file end-to-end.

    Args:
        sessions:  (encoder_sess, decoder_sess) from ``load_sessions()``.
        tokenizer: from ``load_tokenizer()``.
        audio_path: path to an audio file (mp3, wav, etc.).
        chunk_callback: optional callable(info_dict) invoked after each chunk.

    Returns:
        Transcribed text string.
    """
    encoder_sess, decoder_sess = sessions

    audio, sr = load_audio(audio_path)
    audio = resample(audio, sr)

    total_duration_s = len(audio) / SAMPLE_RATE
    chunks = chunk_audio(audio)
    num_chunks = len(chunks)
    all_tokens = []
    cumulative_s = 0.0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_duration_s = len(chunk) / SAMPLE_RATE

        mel = mel_spectrogram(chunk)
        t_wall_start = time.time()
        tokens, timing = transcribe_chunk(
            encoder_sess, decoder_sess, tokenizer, mel, return_timing=True
        )
        wall_time_s = time.time() - t_wall_start

        all_tokens.extend(tokens)
        cumulative_s += chunk_duration_s

        if chunk_callback is not None:
            chunk_callback({
                "chunk_index": chunk_idx,
                "num_chunks": num_chunks,
                "audio_duration_s": total_duration_s,
                "chunk_duration_s": chunk_duration_s,
                "chunk_done_s": cumulative_s,
                "encoder_time_s": timing["encoder_time_s"],
                "decoder_time_s": timing["decoder_time_s"],
                "wall_time_s": wall_time_s,
            })

    return tokenizer.decode(all_tokens).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper Large V3 Turbo on NPU"
    )
    parser.add_argument("files", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--models-dir", default=None,
                        help=f"ONNX models directory (default: {MODELS_DIR})")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading ONNX sessions (QNN)...")
    sessions = load_sessions(args.models_dir)
    print(f"  Sessions loaded in {time.time() - t0:.1f}s")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("  Tokenizer ready")

    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"[!] Not found: {fpath}")
            continue
        print(f"\nTranscribing: {fpath}")
        t1 = time.time()
        text = transcribe(sessions, tokenizer, fpath)
        print(f"  [{time.time() - t1:.1f}s] {text}")


if __name__ == "__main__":
    main()
