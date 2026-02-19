#!/usr/bin/env python3
"""
hidock_gui.py - HiDock P1 GUI application
CustomTkinter-based visual interface for browsing, downloading, converting,
and transcribing recordings from the HiDock P1 USB recorder.
"""

import os
import subprocess
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk

import config
import hidock_p1
import transcribe_npu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def fmt_mmss(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    s = int(seconds)
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def parse_file_datetime(f_info):
    """Parse file_info date+time into a datetime for sorting. Returns epoch on failure."""
    date_s = f_info.get("date", "")
    time_s = f_info.get("time", "")
    dt_str = f"{date_s} {time_s}".strip()
    for fmt in ("%Y/%b/%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return datetime(1970, 1, 1)


# ---------------------------------------------------------------------------
# NPU utilization chart
# ---------------------------------------------------------------------------

class NpuChart(tk.Canvas):
    """Rolling line chart showing NPU utilization percentage per chunk."""

    HISTORY = 60
    BG = "#1a1a2e"
    LINE_COLOR = "#00d4ff"
    FILL_COLOR = "#002244"
    GRID_COLOR = "#2a2a4e"
    LABEL_COLOR = "#667"

    def __init__(self, master, **kwargs):
        kwargs.setdefault("bg", self.BG)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)
        self.data = []
        self.bind("<Configure>", lambda e: self._redraw())

    def add_sample(self, pct):
        self.data.append(max(0.0, min(100.0, pct)))
        if len(self.data) > self.HISTORY:
            self.data = self.data[-self.HISTORY:]
        self._redraw()

    def clear(self):
        self.data.clear()
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pad_l, pad_r, pad_t, pad_b = 30, 6, 6, 6
        cw = w - pad_l - pad_r
        ch = h - pad_t - pad_b

        # Grid lines at 25/50/75/100%
        for pct in (25, 50, 75, 100):
            y = pad_t + ch * (1 - pct / 100)
            self.create_line(pad_l, y, w - pad_r, y, fill=self.GRID_COLOR)
            self.create_text(
                pad_l - 4, y, text=f"{pct}%", anchor="e",
                fill=self.LABEL_COLOR, font=("Consolas", 8)
            )

        if not self.data:
            self.create_text(
                w // 2, h // 2, text="No data", fill=self.LABEL_COLOR,
                font=("Consolas", 10)
            )
            return

        n = len(self.data)
        if n == 1:
            pts = [(pad_l + cw / 2, pad_t + ch * (1 - self.data[0] / 100))]
        else:
            pts = []
            for i, v in enumerate(self.data):
                x = pad_l + (i / (n - 1)) * cw
                y = pad_t + ch * (1 - v / 100)
                pts.append((x, y))

        # Filled area
        fill_pts = [pts[0]] + pts + [pts[-1]]
        fill_pts[0] = (pts[0][0], pad_t + ch)
        fill_pts[-1] = (pts[-1][0], pad_t + ch)
        flat = [c for pt in fill_pts for c in pt]
        self.create_polygon(flat, fill=self.FILL_COLOR, outline="")

        # Line
        if len(pts) >= 2:
            flat_line = [c for pt in pts for c in pt]
            self.create_line(flat_line, fill=self.LINE_COLOR, width=2, smooth=True)


# Column widths shared by header and rows
# col 0=checkbox(28), 1=name(flex), 2=size(72), 3=duration(90), 4=date(160), 5=mode(50), 6=status(40)
_COL_WIDTHS = {2: 72, 3: 90, 4: 160, 5: 50, 6: 40}
_MONO = ("Consolas", 13)


# ---------------------------------------------------------------------------
# File row widget
# ---------------------------------------------------------------------------

class FileRow(ctk.CTkFrame):
    """A single row in the file list representing one recording."""

    def __init__(self, master, file_info, **kwargs):
        super().__init__(master, **kwargs)
        self.file_info = file_info
        self.selected = ctk.BooleanVar(value=False)
        self.downloaded_path = None  # set after download

        self.configure(fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)

        font = ctk.CTkFont(family=_MONO[0], size=_MONO[1])

        self.checkbox = ctk.CTkCheckBox(
            self, variable=self.selected, text="", width=28,
            checkbox_width=20, checkbox_height=20
        )
        self.checkbox.grid(row=0, column=0, padx=(4, 0), pady=2)

        ctk.CTkLabel(
            self, text=file_info["name"], anchor="w", font=font
        ).grid(row=0, column=1, padx=(4, 4), pady=2, sticky="w")

        ctk.CTkLabel(
            self, text=fmt_size(file_info["size"]), anchor="e",
            font=font, width=_COL_WIDTHS[2]
        ).grid(row=0, column=2, padx=4, pady=2)

        ctk.CTkLabel(
            self, text=fmt_duration(file_info["duration"]), anchor="e",
            font=font, width=_COL_WIDTHS[3]
        ).grid(row=0, column=3, padx=4, pady=2)

        date_time = f"{file_info['date']} {file_info['time']}".strip()
        ctk.CTkLabel(
            self, text=date_time, anchor="w",
            font=font, width=_COL_WIDTHS[4]
        ).grid(row=0, column=4, padx=4, pady=2)

        ctk.CTkLabel(
            self, text=file_info["mode"], anchor="w",
            font=font, width=_COL_WIDTHS[5]
        ).grid(row=0, column=5, padx=4, pady=2)

        self.status_label = ctk.CTkLabel(
            self, text="", anchor="e",
            font=ctk.CTkFont(size=12), width=_COL_WIDTHS[6]
        )
        self.status_label.grid(row=0, column=6, padx=(0, 8), pady=2)

    def mark_downloaded(self, path):
        self.downloaded_path = path
        self.status_label.configure(text="[mp3]", text_color="#4CAF50")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class HiDockApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HiDock P1")
        self.geometry("1100x700")
        self.minsize(900, 500)

        # State
        self.dev = None
        self.device_info = None
        self.battery_info = None
        self.files = []
        self.file_rows = []
        self.usb_lock = threading.Lock()
        self.npu_sessions = None   # (encoder, decoder) ONNX sessions
        self.tokenizer = None      # tokenizers.Tokenizer
        self.whisper_loading = False
        self.config = config.load()
        self.sort_key = "date"       # "date" or "duration"
        self.sort_desc = True        # descending by default

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nswe")
        self.sidebar.grid_propagate(False)

        self.title_label = ctk.CTkLabel(
            self.sidebar, text="HiDock P1",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(padx=16, pady=(20, 4))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, text="USB Recording Manager",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.subtitle_label.pack(padx=16, pady=(0, 16))

        self.connect_btn = ctk.CTkButton(
            self.sidebar, text="Connect", command=self._on_connect
        )
        self.connect_btn.pack(padx=16, pady=4, fill="x")

        # Device info area
        self.info_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.info_frame.pack(padx=16, pady=(12, 4), fill="x")

        self.fw_label = ctk.CTkLabel(
            self.info_frame, text="", anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.fw_label.pack(fill="x")

        self.serial_label = ctk.CTkLabel(
            self.info_frame, text="", anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.serial_label.pack(fill="x")

        self.battery_label = ctk.CTkLabel(
            self.info_frame, text="", anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.battery_label.pack(fill="x")

        sep = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        sep.pack(padx=16, pady=12, fill="x")

        self.dl_selected_btn = ctk.CTkButton(
            self.sidebar, text="Download Selected",
            command=self._on_download_selected, state="disabled"
        )
        self.dl_selected_btn.pack(padx=16, pady=4, fill="x")

        self.dl_all_btn = ctk.CTkButton(
            self.sidebar, text="Download All",
            command=self._on_download_all, state="disabled"
        )
        self.dl_all_btn.pack(padx=16, pady=4, fill="x")

        sep2 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        sep2.pack(padx=16, pady=12, fill="x")

        self.transcribe_btn = ctk.CTkButton(
            self.sidebar, text="Transcribe",
            command=self._on_transcribe, state="disabled"
        )
        self.transcribe_btn.pack(padx=16, pady=4, fill="x")

        self.model_label = ctk.CTkLabel(
            self.sidebar, text="Model: not loaded",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.model_label.pack(padx=16, pady=(4, 8))

        sep3 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        sep3.pack(padx=16, pady=12, fill="x")

        self.dl_dir_btn = ctk.CTkButton(
            self.sidebar, text="Download Folder...",
            command=self._on_choose_download_dir
        )
        self.dl_dir_btn.pack(padx=16, pady=4, fill="x")

        self.dl_dir_label = ctk.CTkLabel(
            self.sidebar, text=self._format_config_dir("download_dir"),
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=190
        )
        self.dl_dir_label.pack(padx=16, pady=(2, 4))

        self.output_dir_btn = ctk.CTkButton(
            self.sidebar, text="Transcript Folder...",
            command=self._on_choose_output_dir
        )
        self.output_dir_btn.pack(padx=16, pady=4, fill="x")

        self.output_dir_label = ctk.CTkLabel(
            self.sidebar, text=self._format_config_dir("transcript_output_dir"),
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=190
        )
        self.output_dir_label.pack(padx=16, pady=(2, 8))

        # --- Main panel ---
        self.main_panel = ctk.CTkFrame(self, corner_radius=0)
        self.main_panel.grid(row=0, column=1, sticky="nswe")
        self.main_panel.grid_rowconfigure(0, weight=3)  # file list
        # row 1 = progress (no weight)
        self.main_panel.grid_rowconfigure(2, weight=1)  # NPU chart
        self.main_panel.grid_rowconfigure(3, weight=2)  # transcript
        self.main_panel.grid_columnconfigure(0, weight=1)

        # File list header with column labels + sort buttons
        header_frame = ctk.CTkFrame(self.main_panel, height=30, fg_color="gray20", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="nwe", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)

        hdr_font = ctk.CTkFont(size=12, weight="bold")
        sort_font = ctk.CTkFont(size=11, weight="bold")

        # Checkbox-width spacer
        ctk.CTkLabel(header_frame, text="", width=28).grid(row=0, column=0, padx=(4, 0))

        ctk.CTkLabel(
            header_frame, text="Name", anchor="w", font=hdr_font
        ).grid(row=0, column=1, padx=(4, 4), pady=4, sticky="w")

        ctk.CTkLabel(
            header_frame, text="Size", anchor="e",
            font=hdr_font, width=_COL_WIDTHS[2]
        ).grid(row=0, column=2, padx=4, pady=4)

        self.sort_dur_btn = ctk.CTkButton(
            header_frame, text="Length v", width=_COL_WIDTHS[3], height=22,
            font=sort_font, fg_color="transparent", text_color="white",
            hover_color="gray30", anchor="e",
            command=lambda: self._toggle_sort("duration")
        )
        self.sort_dur_btn.grid(row=0, column=3, padx=4, pady=3)

        self.sort_date_btn = ctk.CTkButton(
            header_frame, text="Date v", width=_COL_WIDTHS[4], height=22,
            font=sort_font, fg_color="transparent", text_color="white",
            hover_color="gray30", anchor="w",
            command=lambda: self._toggle_sort("date")
        )
        self.sort_date_btn.grid(row=0, column=4, padx=4, pady=3)

        ctk.CTkLabel(
            header_frame, text="Mode", anchor="w",
            font=hdr_font, width=_COL_WIDTHS[5]
        ).grid(row=0, column=5, padx=4, pady=4)

        # Status column spacer
        ctk.CTkLabel(header_frame, text="", width=_COL_WIDTHS[6]).grid(
            row=0, column=6, padx=(0, 8)
        )

        # File list scrollable area
        self.file_list_frame = ctk.CTkScrollableFrame(
            self.main_panel, fg_color="transparent"
        )
        self.file_list_frame.grid(row=0, column=0, sticky="nswe", padx=0, pady=(30, 0))
        self.file_list_frame.grid_columnconfigure(0, weight=1)

        # Placeholder when no files
        self.placeholder_label = ctk.CTkLabel(
            self.file_list_frame,
            text="Connect a device to view recordings",
            text_color="gray", font=ctk.CTkFont(size=14)
        )
        self.placeholder_label.pack(pady=40)

        # Progress area
        self.progress_frame = ctk.CTkFrame(self.main_panel, height=50, fg_color="transparent")
        self.progress_frame.grid(row=1, column=0, sticky="we", padx=12, pady=4)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.progress_frame, text="Ready", anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=1, column=0, sticky="we", pady=(2, 0))
        self.progress_bar.set(0)

        # NPU utilization chart
        npu_header = ctk.CTkFrame(self.main_panel, height=30, fg_color="gray20", corner_radius=0)
        npu_header.grid(row=2, column=0, sticky="nwe", padx=0, pady=0)
        npu_header.grid_propagate(False)
        npu_label = ctk.CTkLabel(
            npu_header, text="  NPU Utilization",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold")
        )
        npu_label.pack(fill="x", padx=8, pady=4)

        self.npu_chart = NpuChart(self.main_panel)
        self.npu_chart.grid(row=2, column=0, sticky="nswe", padx=0, pady=(30, 0))

        # Transcription output
        transcript_header = ctk.CTkFrame(self.main_panel, height=30, fg_color="gray20", corner_radius=0)
        transcript_header.grid(row=3, column=0, sticky="nwe", padx=0, pady=0)
        transcript_header.grid_propagate(False)
        transcript_label = ctk.CTkLabel(
            transcript_header, text="  Transcription Output",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold")
        )
        transcript_label.pack(fill="x", padx=8, pady=4)

        self.transcript_box = ctk.CTkTextbox(
            self.main_panel, state="disabled",
            font=ctk.CTkFont(family="Consolas", size=13)
        )
        self.transcript_box.grid(row=3, column=0, sticky="nswe", padx=0, pady=(30, 0))

    # -----------------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------------

    def _set_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))

    def _set_progress(self, value):
        self.after(0, lambda: self.progress_bar.set(value))

    def _set_progress_mode(self, indeterminate=False):
        def _apply():
            if indeterminate:
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start()
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar.set(0)
        self.after(0, _apply)

    def _append_transcript(self, text):
        def _apply():
            self.transcript_box.configure(state="normal")
            self.transcript_box.insert("end", text)
            self.transcript_box.see("end")
            self.transcript_box.configure(state="disabled")
        self.after(0, _apply)

    def _set_buttons_state(self, connected=False, has_downloads=False):
        def _apply():
            state_conn = "normal" if connected else "disabled"
            self.dl_selected_btn.configure(state=state_conn)
            self.dl_all_btn.configure(state=state_conn)
            self.transcribe_btn.configure(state="normal" if has_downloads else "disabled")
        self.after(0, _apply)

    def _populate_device_info(self):
        def _apply():
            if self.device_info:
                self.fw_label.configure(text=f"FW: {self.device_info['version']}")
                self.serial_label.configure(text=f"SN: {self.device_info['serial']}")
            if self.battery_info:
                self.battery_label.configure(
                    text=f"Bat: {self.battery_info['battery']}% ({self.battery_info['status']})"
                )
        self.after(0, _apply)

    def _populate_file_list(self):
        def _apply():
            # Clear existing rows
            for row in self.file_rows:
                row.destroy()
            self.file_rows.clear()
            self.placeholder_label.pack_forget()

            if not self.files:
                self.placeholder_label.pack(pady=40)
                return

            for f_info in self._sorted_files():
                row = FileRow(self.file_list_frame, f_info)
                row.pack(fill="x", padx=4, pady=1)
                self.file_rows.append(row)
        self.after(0, _apply)

    def _toggle_sort(self, key):
        """Toggle sort column / direction and refresh the file list."""
        if self.sort_key == key:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_key = key
            self.sort_desc = True
        self._update_sort_buttons()
        self._populate_file_list()
        self.after(10, self._check_existing_downloads)

    def _update_sort_buttons(self):
        arrow = " v" if self.sort_desc else " ^"
        self.sort_date_btn.configure(
            text="Date" + (arrow if self.sort_key == "date" else "  ")
        )
        self.sort_dur_btn.configure(
            text="Length" + (arrow if self.sort_key == "duration" else "  ")
        )

    def _sorted_files(self):
        """Return self.files sorted by current sort key/direction."""
        if self.sort_key == "date":
            key_fn = parse_file_datetime
        else:
            key_fn = lambda f: f.get("duration", 0)
        return sorted(self.files, key=key_fn, reverse=self.sort_desc)

    def _check_existing_downloads(self):
        """Mark file rows whose mp3 already exists in the download folder."""
        dl_dir = self.config.get("download_dir", "")
        if not dl_dir or not self.file_rows:
            return
        any_found = False
        for row in self.file_rows:
            base = os.path.splitext(row.file_info["name"])[0]
            mp3_path = os.path.join(dl_dir, base + ".mp3")
            if os.path.exists(mp3_path):
                row.mark_downloaded(mp3_path)
                any_found = True
        if any_found:
            self._set_buttons_state(
                connected=self.dev is not None, has_downloads=True
            )

    # -----------------------------------------------------------------------
    # Config / Output folder
    # -----------------------------------------------------------------------

    def _format_config_dir(self, key):
        d = self.config.get(key, "")
        if not d:
            return "No folder set"
        if len(d) > 35:
            return "..." + d[-32:]
        return d

    def _on_choose_download_dir(self):
        initial = self.config.get("download_dir", "") or None
        d = filedialog.askdirectory(title="Choose download folder",
                                    initialdir=initial)
        if d:
            self.config["download_dir"] = d
            config.save(self.config)
            self.dl_dir_label.configure(text=self._format_config_dir("download_dir"))
            self._check_existing_downloads()

    def _on_choose_output_dir(self):
        initial = self.config.get("transcript_output_dir", "") or None
        d = filedialog.askdirectory(title="Choose transcript output folder",
                                    initialdir=initial)
        if d:
            self.config["transcript_output_dir"] = d
            config.save(self.config)
            self.output_dir_label.configure(
                text=self._format_config_dir("transcript_output_dir")
            )

    # -----------------------------------------------------------------------
    # Connect / Disconnect
    # -----------------------------------------------------------------------

    def _on_connect(self):
        if self.dev is not None:
            self._disconnect()
            return
        self.connect_btn.configure(state="disabled", text="Connecting...")
        self._set_status("Connecting to device...")
        self._set_progress_mode(indeterminate=True)
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            with self.usb_lock:
                dev = hidock_p1.open_device()
                info = hidock_p1.get_device_info(dev)
                batt = hidock_p1.get_battery(dev)
                files = hidock_p1.list_files(dev)

            self.dev = dev
            self.device_info = info
            self.battery_info = batt
            self.files = files

            self._populate_device_info()
            self._populate_file_list()
            self._set_buttons_state(connected=True)
            self.after(10, self._check_existing_downloads)
            self._set_status(f"Connected — {len(files)} recording(s)")
            self._set_progress_mode(indeterminate=False)
            self.after(0, lambda: self.connect_btn.configure(
                state="normal", text="Disconnect"
            ))
        except Exception as e:
            self._set_status(f"Connection failed: {e}")
            self._set_progress_mode(indeterminate=False)
            self.after(0, lambda: self.connect_btn.configure(
                state="normal", text="Connect"
            ))

    def _disconnect(self):
        if self.dev is not None:
            try:
                with self.usb_lock:
                    hidock_p1.close_device(self.dev)
            except Exception:
                pass
            self.dev = None
        self.device_info = None
        self.battery_info = None
        self.files = []

        self.fw_label.configure(text="")
        self.serial_label.configure(text="")
        self.battery_label.configure(text="")
        self._populate_file_list()
        self._set_buttons_state(connected=False)
        self.connect_btn.configure(text="Connect")
        self._set_status("Disconnected")

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------

    def _on_download_selected(self):
        selected = [r for r in self.file_rows if r.selected.get()]
        if not selected:
            self._set_status("No files selected")
            return
        self._start_download(selected)

    def _on_download_all(self):
        if not self.file_rows:
            return
        self._start_download(self.file_rows)

    def _start_download(self, rows):
        out_dir = self.config.get("download_dir", "")
        if not out_dir:
            out_dir = filedialog.askdirectory(title="Choose download folder")
            if not out_dir:
                return
        # Disable buttons during download
        self.dl_selected_btn.configure(state="disabled")
        self.dl_all_btn.configure(state="disabled")
        self.connect_btn.configure(state="disabled")
        threading.Thread(
            target=self._download_worker, args=(rows, out_dir), daemon=True
        ).start()

    def _download_worker(self, rows, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        total = len(rows)
        any_downloaded = False

        for idx, row in enumerate(rows, 1):
            f_info = row.file_info
            name = f_info["name"]
            size = f_info["size"]
            hda_path = os.path.join(out_dir, name)

            self._set_status(f"Downloading {idx}/{total}: {name}")
            self._set_progress(0)

            def progress_cb(received, total_size):
                self._set_progress(received / total_size if total_size > 0 else 0)

            try:
                with self.usb_lock:
                    hidock_p1.download_file(
                        self.dev, name, hda_path, size,
                        progress_callback=progress_cb
                    )
            except Exception as e:
                self._set_status(f"Download failed ({name}): {e}")
                continue

            # Convert .hda to .mp3 via ffmpeg
            base, ext = os.path.splitext(name)
            mp3_path = os.path.join(out_dir, base + ".mp3")

            if ext.lower() in (".hda", ".wav"):
                self._set_status(f"Converting {idx}/{total}: {name} -> .mp3")
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-i", hda_path, "-y", mp3_path],
                        capture_output=True, text=True, timeout=120
                    )
                    if result.returncode == 0 and os.path.exists(mp3_path):
                        self.after(0, lambda r=row, p=mp3_path: r.mark_downloaded(p))
                        any_downloaded = True
                    else:
                        self._set_status(
                            f"ffmpeg failed for {name}: {result.stderr[:200]}"
                        )
                except FileNotFoundError:
                    self._set_status("ffmpeg not found — install ffmpeg and add to PATH")
                    break
                except subprocess.TimeoutExpired:
                    self._set_status(f"ffmpeg timed out for {name}")
            else:
                # Non-audio file, just mark as downloaded
                self.after(0, lambda r=row, p=hda_path: r.mark_downloaded(p))
                any_downloaded = True

        self._set_progress(1.0)
        self._set_status(f"Download complete — {total} file(s) processed")
        self.after(0, lambda: self._set_buttons_state(
            connected=self.dev is not None, has_downloads=any_downloaded
        ))
        self.after(0, lambda: self.connect_btn.configure(
            state="normal", text="Disconnect" if self.dev else "Connect"
        ))

    # -----------------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------------

    def _on_transcribe(self):
        downloaded = [r for r in self.file_rows if r.downloaded_path is not None]
        if not downloaded:
            self._set_status("No downloaded files to transcribe")
            return
        self.transcribe_btn.configure(state="disabled")
        threading.Thread(
            target=self._transcribe_worker, args=(downloaded,), daemon=True
        ).start()

    def _load_whisper(self):
        """Lazy-load ONNX sessions + tokenizer for NPU transcription."""
        self.whisper_loading = True
        self._set_progress_mode(indeterminate=True)

        self._set_status("Loading Whisper on NPU (QNN)...")
        self.after(0, lambda: self.model_label.configure(text="Model: loading (NPU)..."))
        try:
            self.npu_sessions = transcribe_npu.load_sessions()
            self.tokenizer = transcribe_npu.load_tokenizer()
            self.after(0, lambda: self.model_label.configure(
                text="Model: loaded (NPU)", text_color="#4CAF50"
            ))
            return True
        except Exception as e:
            self._set_status(f"Failed to load Whisper: {e}")
            self.after(0, lambda: self.model_label.configure(
                text="Model: error", text_color="#F44336"
            ))
            return False

    def _transcribe_file(self, mp3_path, chunk_callback=None):
        """Transcribe a single file on the NPU."""
        return transcribe_npu.transcribe(
            self.npu_sessions, self.tokenizer, mp3_path,
            chunk_callback=chunk_callback
        )

    def _transcribe_worker(self, rows):
        # Lazy-load model on first use
        if self.npu_sessions is None:
            if self.whisper_loading:
                return
            try:
                if not self._load_whisper():
                    self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
                    return
            finally:
                self.whisper_loading = False
                self._set_progress_mode(indeterminate=False)

        self.after(0, self.npu_chart.clear)

        # Compute total audio duration for time-based progress
        total_audio_s = sum(row.file_info["duration"] for row in rows)
        files_done_s = 0.0
        total = len(rows)
        output_dir = self.config.get("transcript_output_dir", "")

        for idx, row in enumerate(rows, 1):
            mp3_path = row.downloaded_path
            filename = os.path.basename(mp3_path)
            last_info = {}

            def on_chunk(info, _files_done=files_done_s):
                nonlocal last_info
                last_info = info
                current_s = _files_done + info["chunk_done_s"]
                pct = current_s / total_audio_s if total_audio_s > 0 else 0
                self._set_progress(pct)
                self._set_status(
                    f"Transcribing {idx}/{total}: {filename} — "
                    f"{fmt_mmss(current_s)} of {fmt_mmss(total_audio_s)}"
                )
                wall = info["wall_time_s"]
                if wall > 0:
                    npu_pct = (info["encoder_time_s"] + info["decoder_time_s"]) / wall * 100
                    self.after(0, lambda p=npu_pct: self.npu_chart.add_sample(p))

            try:
                text = self._transcribe_file(mp3_path, chunk_callback=on_chunk)
                self._append_transcript(f"\n--- {filename} ---\n{text}\n")

                # Save transcript to disk if output dir configured
                if output_dir:
                    base_name = os.path.splitext(filename)[0]
                    txt_path = os.path.join(output_dir, base_name + ".txt")
                    os.makedirs(output_dir, exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
            except Exception as e:
                self._append_transcript(f"\n--- {filename} ---\n[Error: {e}]\n")

            files_done_s += last_info.get("audio_duration_s", row.file_info["duration"])

        self._set_progress(1.0)
        self.after(0, self.npu_chart.clear)
        saved_msg = f" — saved to {output_dir}" if output_dir else ""
        self._set_status(f"Transcription complete — {total} file(s) [NPU]{saved_msg}")
        self.after(0, lambda: self.transcribe_btn.configure(state="normal"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = HiDockApp()
    app.mainloop()
