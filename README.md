# HiDock P1 USB File Access Tool

Desktop tool to access recordings from the HiDock P1 recorder via USB, without using the proprietary HiNotes web software.

## Features

- List all recordings on device
- Download individual files or all files at once
- View device info, battery status, and firmware version

## Setup

### Windows (Native)

1. **Install Python dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Install libusb backend:**
   - Download and install [Zadig](https://zadig.akeo.ie/)
   - Connect your HiDock P1
   - Run Zadig as Administrator
   - Select **Options > List All Devices**
   - Find "HiDock_P1" (VID: 10D6, PID: B00E) in the dropdown
   - Select **WinUSB** or **libusb-win32** as the target driver
   - Click "Replace Driver" (or "Install Driver")
   
   > ⚠️ This replaces the default driver. To restore, use Device Manager to uninstall the device and let Windows reinstall the original driver.

3. **Run:**
   ```powershell
   python hidock_p1.py info
   python hidock_p1.py list
   python hidock_p1.py download "25JAN15-143022-ROOM1.wav"
   python hidock_p1.py download-all ./recordings
   ```

### WSL2 (Linux)

1. **Install dependencies:**
   ```bash
   pip3 install pyusb
   ```

2. **Attach USB device to WSL (run in PowerShell as Admin):**
   ```powershell
   # List USB devices
   usbipd list
   
   # Attach HiDock to WSL (replace X-Y with actual bus ID)
   usbipd attach --wsl --busid X-Y
   ```

3. **Run as root (needed for USB access):**
   ```bash
   sudo python3 hidock_p1.py info
   sudo python3 hidock_p1.py list
   sudo python3 hidock_p1.py download "25JAN15-143022-ROOM1.wav"
   sudo python3 hidock_p1.py download-all ./recordings
   ```

## Usage

```
hidock_p1.py info              Show device info and battery
hidock_p1.py list              List all recordings on device
hidock_p1.py download <file>   Download a single file
hidock_p1.py download-all      Download all recordings
```

## Protocol Notes

- **Vendor:** Actions Semiconductor (VID 0x10D6)
- **Product:** HiDock_P1 (PID 0xB00E)
- **Packet format:** 12-byte header + body
  - Bytes 0-1: Magic (0x12 0x34)
  - Bytes 2-3: Command ID (uint16 BE)
  - Bytes 4-7: Sequence number (uint32 BE)
  - Bytes 8-11: Length field (padding in top byte, body length in lower 24 bits)

## Troubleshooting

- **"HiDock P1 not found"** - Make sure the device is connected and the correct driver is installed (Zadig on Windows, or attached via usbipd on WSL2)
- **Downloads garbled** - The download code assumes each chunk is wrapped in a 12-byte packet header. If files are corrupt, the protocol may send raw bytes after the first ACK.
