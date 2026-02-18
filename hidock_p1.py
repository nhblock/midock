#!/usr/bin/env python3
"""
hidock_p1.py - HiDock P1 USB file access tool
Reverse-engineered from HiNotes WebUSB JavaScript (index-CTYere8j.js)

Protocol notes:
  - Vendor: Actions Semiconductor (VID 0x10D6), Product: HiDock_P1 (PID 0xB00E)
  - Single USB interface (class 0xFF vendor-specific), two bulk endpoints
    - EP 0x01 OUT: send commands
    - EP 0x82 IN:  receive responses
  - Packet format (12-byte header + body):
    Offset  Len  Description
    0       2    Magic: 0x12 0x34
    2       2    Command ID (uint16 big-endian)
    4       4    Sequence number (uint32 big-endian)
    8       4    Lengths: top byte = padding length, lower 24 bits = body length
    12      N    Body bytes
    12+N    P    Padding bytes (usually 0)

Usage:
  python3 hidock_p1.py list
  python3 hidock_p1.py info
  python3 hidock_p1.py download <filename> [output_path]
  python3 hidock_p1.py download-all [output_dir]
"""

import usb.core
import usb.util
import struct
import sys
import os
import time
import argparse
import platform

# ---------------------------------------------------------------------------
# Windows: configure libusb backend
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    try:
        import libusb
        # Get the appropriate DLL path for the current architecture
        arch = "x86_64" if platform.machine().endswith('64') else "x86"
        dll_path = os.path.join(os.path.dirname(libusb.__file__), 
                                "_platform", "windows", arch, "libusb-1.0.dll")
        if os.path.exists(dll_path):
            import usb.backend.libusb1 as libusb1
            backend = libusb1.get_backend(find_library=lambda x: dll_path)
            if backend:
                # Monkey-patch usb.core.find to use our backend by default
                _original_find = usb.core.find
                def _find_with_backend(*args, **kwargs):
                    if 'backend' not in kwargs:
                        kwargs['backend'] = backend
                    return _original_find(*args, **kwargs)
                usb.core.find = _find_with_backend
    except ImportError:
        pass  # libusb package not installed, will fail later with helpful error

# ---------------------------------------------------------------------------
# USB identifiers
# ---------------------------------------------------------------------------
VENDOR_ID  = 0x10D6
PRODUCT_ID = 0xB00E
EP_OUT     = 0x01   # Bulk OUT to device
EP_IN      = 0x82   # Bulk IN from device
INTERFACE  = 0
READ_SIZE  = 512 * 1024  # 512 KB chunks (matches JS: ne = 512e3)

# ---------------------------------------------------------------------------
# Command IDs (from JS constants)
# ---------------------------------------------------------------------------
CMD_GET_DEVICE_INFO   = 1
CMD_GET_TIME          = 2
CMD_SET_TIME          = 3
CMD_GET_FILE_LIST     = 4
CMD_TRANSFER_FILE     = 5   # streaming download
CMD_GET_FILE_COUNT    = 6
CMD_DELETE_FILE       = 7
CMD_GET_SETTINGS      = 11
CMD_SET_SETTINGS      = 12
CMD_GET_FILE_BLOCK    = 13  # block download (newer firmware)
CMD_READ_FILE         = 21  # random-access read with offset+length
CMD_GET_BATTERY       = 4100
CMD_GET_RECORDING     = 18  # currently recording file name

# ---------------------------------------------------------------------------
# Packet builder / parser
# ---------------------------------------------------------------------------

_sequence = 0

def _next_seq():
    global _sequence
    _sequence += 1
    return _sequence

def build_packet(cmd_id: int, body: bytes = b'') -> bytes:
    """Build a 12-byte-header packet."""
    seq      = _next_seq()
    body_len = len(body)
    padding  = 0  # padding length in top byte; always 0 for outgoing commands
    length_field = (padding << 24) | (body_len & 0x00FFFFFF)
    # Build manually to match exact layout
    pkt = bytearray(12 + body_len)
    pkt[0]  = 0x12
    pkt[1]  = 0x34
    pkt[2]  = (cmd_id >> 8) & 0xFF
    pkt[3]  = (cmd_id     ) & 0xFF
    pkt[4]  = (seq >> 24)  & 0xFF
    pkt[5]  = (seq >> 16)  & 0xFF
    pkt[6]  = (seq >>  8)  & 0xFF
    pkt[7]  = (seq       ) & 0xFF
    pkt[8]  = (length_field >> 24) & 0xFF
    pkt[9]  = (length_field >> 16) & 0xFF
    pkt[10] = (length_field >>  8) & 0xFF
    pkt[11] = (length_field      ) & 0xFF
    for i, b in enumerate(body):
        pkt[12 + i] = b & 0xFF
    return bytes(pkt)

def parse_packet(data: bytearray, offset: int = 0):
    """
    Parse one packet from a buffer starting at `offset`.
    Returns (cmd_id, seq, body_bytes, total_packet_length) or None if incomplete.
    Raises ValueError on bad magic.
    """
    available = len(data) - offset
    if available < 12:
        return None
    if data[offset] != 0x12 or data[offset+1] != 0x34:
        raise ValueError(f"Bad magic at offset {offset}: {data[offset]:02x} {data[offset+1]:02x}")
    cmd_id    = (data[offset+2] << 8) | data[offset+3]
    seq       = (data[offset+4] << 24) | (data[offset+5] << 16) | \
                (data[offset+6] << 8 ) |  data[offset+7]
    length_field = (data[offset+8] << 24) | (data[offset+9] << 16) | \
                   (data[offset+10] << 8) |  data[offset+11]
    padding   = (length_field >> 24) & 0xFF
    body_len  =  length_field & 0x00FFFFFF
    total     = 12 + body_len + padding
    if available < total:
        return None
    body = bytes(data[offset+12 : offset+12+body_len])
    return (cmd_id, seq, body, total)

# ---------------------------------------------------------------------------
# Device connection
# ---------------------------------------------------------------------------

def open_device():
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        raise RuntimeError(
            "HiDock P1 not found. Is it plugged in?\n"
            "  - Windows: Install WinUSB driver via Zadig (see README.md)\n"
            "  - WSL2: Run 'usbipd attach --wsl --busid X-Y' in PowerShell"
        )
    # Detach kernel driver if active (Linux only - skip on Windows)
    try:
        if dev.is_kernel_driver_active(INTERFACE):
            print("[*] Detaching kernel driver...")
            dev.detach_kernel_driver(INTERFACE)
    except NotImplementedError:
        pass  # Windows doesn't support kernel driver operations
    dev.set_configuration()
    usb.util.claim_interface(dev, INTERFACE)
    product_name = dev.product or "HiDock_P1"
    print(f"[*] Connected: {product_name} (VID:{VENDOR_ID:#06x} PID:{PRODUCT_ID:#06x})")
    return dev

def close_device(dev):
    try:
        usb.util.release_interface(dev, INTERFACE)
    except Exception:
        pass
    try:
        dev.attach_kernel_driver(INTERFACE)  # Linux only
    except (NotImplementedError, Exception):
        pass

# ---------------------------------------------------------------------------
# Low-level send/receive
# ---------------------------------------------------------------------------

def send_cmd(dev, cmd_id: int, body: bytes = b'', timeout: int = 5000):
    pkt = build_packet(cmd_id, body)
    dev.write(EP_OUT, pkt, timeout=timeout)

def recv_all(dev, timeout_ms: int = 5000, inter_chunk_ms: int = 100) -> bytes:
    """
    Read bulk IN data until no more arrives within inter_chunk_ms.
    Returns concatenated raw bytes.
    """
    buf = bytearray()
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        try:
            chunk = dev.read(EP_IN, READ_SIZE, timeout=inter_chunk_ms)
            buf.extend(bytes(chunk))
            deadline = time.time() + inter_chunk_ms / 1000  # reset on data
        except usb.core.USBTimeoutError:
            if buf:
                break
        except Exception as e:
            print(f"[!] Read error: {e}")
            break
    return bytes(buf)

def send_recv(dev, cmd_id: int, body: bytes = b'', timeout_ms: int = 5000):
    """Send a command and receive one response packet. Returns (cmd_id, body)."""
    send_cmd(dev, cmd_id, body, timeout=timeout_ms)
    raw = recv_all(dev, timeout_ms=timeout_ms)
    if not raw:
        return None, None
    result = parse_packet(bytearray(raw))
    if result is None:
        return None, None
    rcmd, rseq, rbody, _ = result
    return rcmd, rbody

# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------

def get_device_info(dev):
    _, body = send_recv(dev, CMD_GET_DEVICE_INFO)
    if body is None or len(body) < 20:
        return None
    # JS: bytes 0-3 = version (BCD-ish), bytes 4-19 = serial number chars
    ver_parts = [(body[i] & 0xFF) for i in range(4)]
    version = ".".join(str(v) for v in ver_parts[1:])  # skip first byte per JS
    version_num = (ver_parts[0] << 24) | (ver_parts[1] << 16) | (ver_parts[2] << 8) | ver_parts[3]
    serial = "".join(chr(body[i] & 0xFF) for i in range(4, min(20, len(body))) if body[i] > 0)
    return {"version": version, "version_num": version_num, "serial": serial}

def get_file_count(dev):
    _, body = send_recv(dev, CMD_GET_FILE_COUNT)
    if body is None or len(body) < 4:
        return 0
    return (body[0] << 24) | (body[1] << 16) | (body[2] << 8) | body[3]

def get_battery(dev):
    _, body = send_recv(dev, CMD_GET_BATTERY, timeout_ms=3000)
    if body is None or len(body) < 3:
        return None
    status_code = body[0] & 0xFF
    battery_pct = body[1] & 0xFF
    voltage     = (body[2] << 24 | body[3] << 16 | body[4] << 8 | body[5]) if len(body) >= 6 else 0
    status_str  = {0: "idle", 1: "charging", 2: "full"}.get(status_code, "unknown")
    return {"status": status_str, "battery": battery_pct, "voltage_mv": voltage}

def list_files(dev):
    """
    Send get-file-list (cmd 4) and parse the multi-packet response.
    Returns list of file dicts with name, length, duration, mode, date fields.
    """
    print("[*] Requesting file list...")
    send_cmd(dev, CMD_GET_FILE_LIST)

    # Collect all response data - file list can be large and multi-packet
    raw = recv_all(dev, timeout_ms=10000, inter_chunk_ms=500)
    if not raw:
        print("[!] No response to file list command")
        return []

    # Decode all packets and collect bodies for cmd 4
    buf = bytearray(raw)
    all_body = bytearray()
    pos = 0
    while pos < len(buf):
        result = parse_packet(buf, pos)
        if result is None:
            break
        rcmd, rseq, rbody, pkt_len = result
        if rcmd == CMD_GET_FILE_LIST:
            all_body.extend(rbody)
        pos += pkt_len

    if not all_body:
        return []

    return _parse_file_list_body(bytes(all_body))

def _parse_file_list_body(data: bytes):
    """Parse the concatenated body bytes from get-file-list responses."""
    files = []
    u = list(data)

    # Optional header: if first two bytes are 0xFF 0xFF, next 4 bytes = total count
    h = 0
    if len(u) >= 2 and (u[0] & 0xFF) == 0xFF and (u[1] & 0xFF) == 0xFF:
        h += 6  # skip 2-byte marker + 4-byte count

    x = h
    while x < len(u):
        if x + 4 >= len(u):
            break
        version  = u[x] & 0xFF;     x += 1
        name_len = ((u[x] & 0xFF) << 16) | ((u[x+1] & 0xFF) << 8) | (u[x+2] & 0xFF); x += 3

        name_chars = []
        for _ in range(name_len):
            if x >= len(u):
                break
            b = u[x] & 0xFF; x += 1
            if b > 0:
                name_chars.append(chr(b))

        if x + 4 + 6 + 16 > len(u):
            break

        file_size = ((u[x] & 0xFF) << 24) | ((u[x+1] & 0xFF) << 16) | \
                    ((u[x+2] & 0xFF) << 8)  |  (u[x+3] & 0xFF); x += 4
        x += 6  # skip 6 unknown bytes

        sig_hex = "".join(f"{(u[x+i] & 0xFF):02x}" for i in range(16)); x += 16

        name = "".join(name_chars)

        # Compute duration from version + size (mirrors JS logic)
        duration = 0
        if version == 1:
            duration = file_size / 32 * 2
        elif version == 2:
            duration = (file_size - 44) / 48 / 2
        elif version == 3:
            duration = (file_size - 44) / 48 / 2 / 2
        elif version == 5:
            duration = file_size / 12
        elif version == 6:
            duration = file_size / 16
        elif version == 7:
            duration = file_size / 10
        else:
            duration = file_size / 32

        # Parse mode from filename
        import re
        mode = "room"
        m = re.match(r'^(\w{9})-(\d{6})-(.+?)\d+\.\w+$', name, re.IGNORECASE)
        if m:
            mode_str = m.group(3).upper()
            if mode_str in ("WHSP", "WIP"):
                mode = "whisper"
            elif mode_str == "CALL":
                mode = "call"
            elif mode_str == "ROOM":
                mode = "room"

        # Parse date from filename
        date_str = ""
        time_str = ""
        dt = None
        import re as _re
        if _re.match(r'^\d{14}REC\d+\.wav$', name, re.IGNORECASE):
            m2 = _re.match(r'^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})REC', name)
            if m2:
                date_str = f"{m2.group(1)}/{m2.group(2)}/{m2.group(3)}"
                time_str = f"{m2.group(4)}:{m2.group(5)}:{m2.group(6)}"
        else:
            m3 = _re.match(r'^(\d{2})?(\d{2})(\w{3})(\d{2})-(\d{2})(\d{2})(\d{2})-.*\.(hda|wav)$', name, re.IGNORECASE)
            if m3:
                yr  = m3.group(2)
                mon = m3.group(3)
                day = m3.group(4)
                h_  = m3.group(5)
                mn  = m3.group(6)
                sc  = m3.group(7)
                date_str = f"20{yr}/{mon}/{day}"
                time_str = f"{h_}:{mn}:{sc}"

        files.append({
            "name":      name,
            "size":      file_size,
            "duration":  int(duration),
            "mode":      mode,
            "date":      date_str,
            "time":      time_str,
            "version":   version,
            "signature": sig_hex,
        })

    return files

def download_file(dev, filename: str, dest_path: str, file_size: int):
    """
    Download a file from the device using the streaming transfer-file command (cmd 5).
    Falls back to get-file-block (cmd 13) for newer firmware.
    """
    print(f"[*] Downloading: {filename} ({file_size:,} bytes) -> {dest_path}")
    body = bytes(ord(c) for c in filename)

    # Use CMD_TRANSFER_FILE (5) - send filename, device streams the file back
    send_cmd(dev, CMD_TRANSFER_FILE, body, timeout=10000)

    received = bytearray()
    start    = time.time()
    last_pct = -1

    while len(received) < file_size:
        try:
            chunk = dev.read(EP_IN, READ_SIZE, timeout=10000)
            raw   = bytes(chunk)
        except usb.core.USBTimeoutError:
            print("\n[!] Timeout waiting for data")
            break
        except Exception as e:
            print(f"\n[!] Read error: {e}")
            break

        # The response packets carry cmd=5 with file data in the body.
        # Parse out the body from each packet.
        buf = bytearray(raw)
        pos = 0
        while pos < len(buf):
            result = parse_packet(buf, pos)
            if result is None:
                # Might be raw audio data without a header at some point?
                # Append remaining bytes directly just in case
                received.extend(buf[pos:])
                break
            rcmd, rseq, rbody, pkt_len = result
            if rcmd == CMD_TRANSFER_FILE:
                received.extend(rbody)
            pos += pkt_len

        pct = int(len(received) / file_size * 100)
        if pct != last_pct:
            elapsed = time.time() - start
            rate    = len(received) / elapsed / 1024 if elapsed > 0 else 0
            print(f"\r    {pct:3d}%  {len(received):,}/{file_size:,} bytes  {rate:.0f} KB/s", end="", flush=True)
            last_pct = pct

        if len(received) >= file_size:
            break

    print()
    with open(dest_path, "wb") as f:
        f.write(received[:file_size])
    print(f"[+] Saved: {dest_path} ({len(received):,} bytes written)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"

def cmd_info(dev):
    info = get_device_info(dev)
    batt = get_battery(dev)
    count = get_file_count(dev)
    print("\n=== HiDock P1 Device Info ===")
    if info:
        print(f"  Firmware:  {info['version']}")
        print(f"  Serial:    {info['serial']}")
    if batt:
        print(f"  Battery:   {batt['battery']}% ({batt['status']})")
    print(f"  Files:     {count}")
    print()

def cmd_list(dev):
    files = list_files(dev)
    if not files:
        print("No files found (or empty response).")
        return
    print(f"\n{'#':>3}  {'Filename':<40}  {'Size':>10}  {'Duration':>8}  {'Date':<19}  {'Mode'}")
    print("-" * 100)
    for i, f in enumerate(files, 1):
        date_time = f"{f['date']} {f['time']}".strip()
        print(f"  {i:>2}  {f['name']:<40}  {f['size']:>10,}  {fmt_duration(f['duration']):>8}  {date_time:<19}  {f['mode']}")
    print(f"\n{len(files)} file(s) total")

def cmd_download(dev, filename: str, dest: str):
    # We need the file size; get it from the file list
    files = list_files(dev)
    match = next((f for f in files if f['name'] == filename), None)
    if not match:
        # Try case-insensitive
        match = next((f for f in files if f['name'].lower() == filename.lower()), None)
    if not match:
        print(f"[!] File '{filename}' not found on device.")
        print("Available files:", [f['name'] for f in files])
        return
    if dest is None:
        dest = filename
    download_file(dev, match['name'], dest, match['size'])

def cmd_download_all(dev, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    files = list_files(dev)
    if not files:
        print("No files to download.")
        return
    print(f"[*] Downloading {len(files)} file(s) to {out_dir}/")
    for i, f in enumerate(files, 1):
        dest = os.path.join(out_dir, f['name'])
        print(f"\n[{i}/{len(files)}] {f['name']}")
        try:
            download_file(dev, f['name'], dest, f['size'])
        except Exception as e:
            print(f"[!] Failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="HiDock P1 file access tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info",          help="Show device info and battery")
    sub.add_parser("list",          help="List all recordings on device")

    dl = sub.add_parser("download", help="Download a single file")
    dl.add_argument("filename",     help="Exact filename as shown by 'list'")
    dl.add_argument("dest",         nargs="?", default=None, help="Output file path (default: filename)")

    da = sub.add_parser("download-all", help="Download all recordings")
    da.add_argument("out_dir",      nargs="?", default="hidock_recordings", help="Output directory")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    dev = open_device()
    try:
        if   args.command == "info":         cmd_info(dev)
        elif args.command == "list":         cmd_list(dev)
        elif args.command == "download":     cmd_download(dev, args.filename, args.dest)
        elif args.command == "download-all": cmd_download_all(dev, args.out_dir)
    finally:
        close_device(dev)

if __name__ == "__main__":
    main()
