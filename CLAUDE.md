# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Desktop CLI tool to access recordings from the HiDock P1 recorder via USB, bypassing the proprietary HiNotes web app. The USB protocol was reverse-engineered from the HiNotes WebUSB JavaScript bundle (stored in `mfr/` for reference).

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Commands (Windows - requires Zadig USB driver setup first)
python hidock_p1.py info
python hidock_p1.py list
python hidock_p1.py download <filename> [output_path]
python hidock_p1.py download-all [output_dir]

# WSL2 - requires usbipd attach and sudo
sudo python3 hidock_p1.py list
```

There is no build step, test suite, linter, or CI/CD pipeline.

## Architecture

Single-file Python CLI (`hidock_p1.py`, ~544 lines) with a layered procedural design:

1. **Platform setup** (top of file) — Monkey-patches `usb.core.find` on Windows to inject the libusb backend DLL path
2. **Constants** — USB identifiers (`VID 0x10D6`, `PID 0xB00E`), endpoint addresses (`EP_OUT=0x01`, `EP_IN=0x82`), and 13 command IDs (`CMD_GET_DEVICE_INFO`, `CMD_GET_FILE_LIST`, `CMD_TRANSFER_FILE`, etc.)
3. **Packet layer** — `build_packet()` / `parse_packet()` handle the 12-byte header format (magic `0x1234`, command ID, sequence number, padding+length field)
4. **Transport layer** — `open_device()` / `close_device()`, `send_cmd()` / `recv_all()` / `send_recv()`. The receive loop uses a deadline-reset strategy: each received USB chunk resets an inter-chunk timeout
5. **Command layer** — `get_device_info()`, `get_battery()`, `list_files()`, `download_file()` etc. File list parsing (`_parse_file_list_body`) handles binary fields: version byte, name, size, signature, and computes duration from a version-dependent formula
6. **CLI layer** — `argparse` subcommands dispatching to `cmd_info()`, `cmd_list()`, `cmd_download()`, `cmd_download_all()`

## Key Implementation Details

- Dependencies are only `pyusb` (cross-platform) and `libusb` (Windows backend DLL provider)
- All USB communication is raw bulk transfer on a vendor-specific interface (class 0xFF)
- File downloads stream chunk-by-chunk, each wrapped in a 12-byte packet header that must be stripped
- The `mfr/` directory contains the original and prettified HiNotes JavaScript — use these as reference when investigating protocol behavior
- Test `.hda` and `.mp3` files in the root are manual test artifacts from a physical device
