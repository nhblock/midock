# Lazy audio conversion at transcribe time

Date: 2026-04-22
Status: Approved

## Problem

The transcription backend (`transcribe_npu.load_audio`) uses the `audio2numpy` library, which supports only `mp3`, `wav`, and `aiff`. When a user drops an `.m4a` file (e.g. an iPhone voice memo) into the Downloads folder and hits **Transcribe**, it surfaces in the list via `scan_audio_files` (because `.m4a` is already in `AUDIO_EXTENSIONS`) but fails at transcribe time:

```
--- New Recording 28.m4a ---
[Error: Only support mp3, wav, aiff formats.]
```

The same failure affects every other non-{mp3,wav} extension currently listed in `AUDIO_EXTENSIONS`: `.flac`, `.aac`, `.ogg`, `.wma`.

## Goal

Allow users to transcribe any audio format in `AUDIO_EXTENSIONS` by converting unsupported formats to `.mp3` via `ffmpeg` on demand, without changing the visible file list or adding UI controls.

## Non-goals

- No changes to `AUDIO_EXTENSIONS`, `scan_audio_files`, `_refresh_local_view`, `_refresh_device_view`, or `_check_existing_downloads`.
- No new UI (no "Convert to MP3" button, no menu item).
- No changes to the HiDock device-download pipeline (`_download_worker`) — it continues to convert `.hda`/`.wav` on download for storage normalization reasons unrelated to this change.
- No change to `transcribe_npu.load_audio` — we satisfy its format constraint by feeding it an `.mp3`.

## Design

### New helper: `ensure_mp3(src_path, status_prefix)`

Module-level function in `hidock_gui.py`. Signature:

```python
def ensure_mp3(src_path: str, status_prefix: str = "") -> str
```

Behavior:

1. Compute `ext = os.path.splitext(src_path)[1].lower()`.
2. If `ext in {".mp3", ".wav"}` → return `src_path` unchanged. (Both are natively supported by `audio2numpy`.)
3. Compute `mp3_path = os.path.splitext(src_path)[0] + ".mp3"` (sibling file in the same directory).
4. If `os.path.exists(mp3_path)` → return `mp3_path`. (Cached from prior run or the HiDock pipeline.)
5. Otherwise invoke `ffmpeg -i <src_path> -y <mp3_path>` with `subprocess.run(..., capture_output=True, text=True, timeout=120)`, mirroring the existing HiDock flow at `hidock_gui.py:1275`.
   - On `FileNotFoundError` → raise `RuntimeError("ffmpeg not found — install ffmpeg and add to PATH")`.
   - On `subprocess.TimeoutExpired` → raise `RuntimeError(f"ffmpeg timed out converting {basename}")`.
   - On non-zero return code → raise `RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")`.
   - On success → return `mp3_path`.

### Call site: `_transcribe_worker`

At `hidock_gui.py:1478`, inside the `for idx, row in enumerate(rows, 1):` loop, before the existing `try:` block on line 1482:

```python
for idx, row in enumerate(rows, 1):
    src_path = row.downloaded_path
    filename = os.path.basename(src_path)

    try:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in (".mp3", ".wav"):
            self._set_status(f"Converting {idx}/{total}: {filename} -> .mp3")
            mp3_path = ensure_mp3(src_path)
            row.downloaded_path = mp3_path  # update in place
            filename = os.path.basename(mp3_path)
        else:
            mp3_path = src_path

        if use_diarize:
            ...
```

The existing `try/except Exception as e` at line 1506 catches any `RuntimeError` from `ensure_mp3` and renders a `[Error: ...]` line in the transcript panel — no separate error path needed.

Mutating `row.downloaded_path` in place is intentional: subsequent re-transcribes of the same row reuse the cached mp3, and `_mark_file_transcribed` records the mp3 path rather than the original m4a.

### Preserve original

The source file (e.g. `New Recording 28.m4a`) is never deleted. The converted sibling (`New Recording 28.mp3`) is created alongside. Matches the `.hda` / `.wav` pattern in the HiDock download flow.

### Scope of extensions converted

Any extension in `AUDIO_EXTENSIONS` that is not `.mp3` and not `.wav`. In practice today that's `.m4a`, `.flac`, `.aac`, `.ogg`, `.wma`. `.hda` never reaches this path because the HiDock download pipeline always converts it to `.mp3` before the row is exposed for transcribe.

## Risks and tradeoffs

- **First-run latency per batch.** A batch of N non-mp3 files adds an ffmpeg pass per file before transcription starts. ffmpeg is fast (seconds per file) and the per-file status shows progress. Second-run is free (cached mp3).
- **Concurrent transcribe of the same row.** Not a concern — `_transcribe_worker` is invoked from a single thread guarded by the Transcribe button's disabled state.
- **`_check_existing_downloads` still scans for `base + ".mp3"`.** That logic is unchanged; after the first transcribe run, the sibling mp3 exists and will be detected on subsequent device views. No regression.

## Testing

Manual verification:

1. Drop `New Recording 28.m4a` into the Downloads folder, click Downloads, select it, click Transcribe. Expect: "Converting 1/1 ... -> .mp3" status, followed by normal transcription, with `New Recording 28.mp3` created alongside the `.m4a`.
2. Re-transcribe the same row. Expect: no conversion step (cached mp3 reused), transcription proceeds immediately.
3. Transcribe a `.wav` file. Expect: no conversion step (passes through directly).
4. Transcribe with ffmpeg removed from PATH. Expect: `[Error: ffmpeg not found — install ffmpeg and add to PATH]` in the transcript panel, transcribe button re-enables.
5. Batch-transcribe mixed `.mp3` + `.m4a` files. Expect: mp3s transcribe immediately; m4as show conversion status then transcribe.
