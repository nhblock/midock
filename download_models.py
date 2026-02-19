#!/usr/bin/env python3
"""
download_models.py - Download ONNX diarization models from HuggingFace.

Downloads:
  1. pyannote_segmentation.onnx (~5 MB) - speaker segmentation
  2. wespeaker_ecapa_tdnn.onnx (~26 MB) - speaker embedding

Models are placed in the models/ directory alongside existing Whisper models.
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    "pyannote_segmentation.onnx": {
        "url": "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx",
        "description": "pyannote-segmentation-3.0 (speaker segmentation)",
        "size_mb": 5,
    },
    "wespeaker_ecapa_tdnn.onnx": {
        "url": "https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/voxceleb_ECAPA512_LM.onnx",
        "description": "wespeaker ECAPA-TDNN-512-LM (speaker embedding)",
        "size_mb": 25,
    },
}


def download_file(url, dest_path, description=""):
    """Download a file with progress reporting."""
    print(f"  Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
        else:
            mb = downloaded / (1024 * 1024)
            print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Model directory: {MODELS_DIR}\n")

    all_ok = True
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename

        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[OK] {filename} already exists ({size_mb:.1f} MB)")
            continue

        print(f"[DL] {filename} (~{info['size_mb']} MB)")
        ok = download_file(info["url"], dest, info["description"])
        if ok:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  Saved: {size_mb:.1f} MB")
        else:
            all_ok = False
            print(f"  FAILED to download {filename}")

    print()
    if all_ok:
        print("All diarization models ready.")
    else:
        print("Some downloads failed. Re-run this script to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
