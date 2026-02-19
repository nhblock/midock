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
import diarize
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

# ---------------------------------------------------------------------------
# Color palette (Vertdure)
# ---------------------------------------------------------------------------

CLR_GREEN        = "#61BF36"   # primary buttons, accents
CLR_GREEN_DARK   = "#3A591B"   # sidebar bg, section headers
CLR_GREEN_DEEP   = "#1e2e12"   # main panel bg
CLR_YELLOW       = "#F2E205"   # highlight accent
CLR_GOLD         = "#F2CB05"   # secondary accent
CLR_AMBER        = "#D9961A"   # delete / warning
CLR_BG_DARK      = "#141c0c"   # darkest background
CLR_BG_PANEL     = "#1a2410"   # main panel background
CLR_HEADER       = "#2a3a18"   # section header bars
CLR_SEP          = "#3A591B"   # separator lines
CLR_TEXT         = "#e8f0e0"   # primary text
CLR_TEXT_DIM     = "#8a9a78"   # secondary / muted text
CLR_TEXT_BRIGHT  = "#ffffff"   # bright text
CLR_RED          = "#c04030"   # error text


class ProcessorChart(tk.Canvas):
    """Scrolling bar-chart activity monitor for NPU and CPU inference steps.

    Fixed-width bars anchored to the right edge — new bars scroll in from
    the right and old ones exit on the left, giving a continuous EKG feel.
    Supports two series: "npu" (green) and "cpu" (blue).
    """

    BAR_W = 4              # fixed px width per bar
    BAR_GAP = 1            # px between bars
    BG = CLR_BG_DARK
    # NPU colors
    BAR_NPU = CLR_GREEN
    BAR_NPU_ENCODER = CLR_YELLOW
    BAR_NPU_DIM = "#2a4018"
    # CPU colors
    BAR_CPU = "#4a9eff"
    BAR_CPU_DIM = "#1a3a5c"
    REDRAW_MS = 40         # throttle: max ~25 fps

    def __init__(self, master, **kwargs):
        kwargs.setdefault("bg", self.BG)
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("height", 40)
        super().__init__(master, **kwargs)
        # Each sample: (time_ms, source, is_encoder)
        # source: "npu" or "cpu"
        self.data = []
        self._redraw_pending = False
        self.bind("<Configure>", lambda e: self._schedule_redraw())

    def add_sample(self, time_ms, source="npu", is_encoder=False):
        """Add a timing sample.

        Args:
            time_ms: inference step duration in milliseconds.
            source: "npu" or "cpu".
            is_encoder: True for encoder steps (full-height yellow spike, NPU only).
        """
        self.data.append((max(0.01, time_ms), source, is_encoder))
        # Trim to max bars that could ever fit on screen
        max_bars = max(400, self.winfo_width() // (self.BAR_W + self.BAR_GAP) + 10)
        if len(self.data) > max_bars:
            self.data = self.data[-max_bars:]
        self._schedule_redraw()

    def clear(self):
        self.data.clear()
        self._schedule_redraw()

    def _schedule_redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after(self.REDRAW_MS, self._do_redraw)

    def _do_redraw(self):
        self._redraw_pending = False
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4 or not self.data:
            return

        pad = 2
        cw = w - pad * 2
        ch = h - pad * 2
        step = self.BAR_W + self.BAR_GAP
        max_visible = cw // step

        # Take only the most recent bars that fit
        visible = self.data[-max_visible:] if len(self.data) > max_visible else self.data
        n = len(visible)
        if n == 0:
            return

        # Separate rolling max per source for normalization
        npu_times = [t for t, src, enc in visible if src == "npu" and not enc]
        cpu_times = [t for t, src, enc in visible if src == "cpu"]
        max_npu_ms = max(npu_times) if npu_times else 1.0
        max_cpu_ms = max(cpu_times) if cpu_times else 1.0

        # Draw right-aligned: newest bar flush with the right edge
        right_edge = pad + cw
        y_bottom = pad + ch

        for i, (ms, source, is_enc) in enumerate(visible):
            x1 = right_edge - (n - 1 - i) * step
            x0 = x1 - self.BAR_W
            age = (n - 1 - i) / max(n - 1, 1)

            if source == "npu":
                if is_enc:
                    bar_h = ch
                    color = self.BAR_NPU_ENCODER
                else:
                    frac = ms / max_npu_ms if max_npu_ms > 0 else 0.5
                    bar_h = max(ch * 0.15, ch * frac)
                    color = self.BAR_NPU_DIM if age > 0.7 else self.BAR_NPU
            else:  # cpu
                frac = ms / max_cpu_ms if max_cpu_ms > 0 else 0.5
                bar_h = max(ch * 0.15, ch * frac)
                color = self.BAR_CPU_DIM if age > 0.7 else self.BAR_CPU

            y0 = y_bottom - bar_h
            self.create_rectangle(x0, y0, x1, y_bottom, fill=color, outline="")


# Column widths shared by header and rows
# col 0=checkbox(28), 1=name(flex), 2=size(72), 3=duration(90), 4=date(160), 5=mode(62), 6=dl(36)
_COL_WIDTHS = {2: 72, 3: 90, 4: 160, 5: 62, 6: 36}
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

        self.dl_label = ctk.CTkLabel(
            self, text="", anchor="center",
            font=ctk.CTkFont(size=12), width=_COL_WIDTHS[6]
        )
        self.dl_label.grid(row=0, column=6, padx=(0, 8), pady=2)

    def mark_downloaded(self, path):
        self.downloaded_path = path
        self.dl_label.configure(text="[yes]", text_color=CLR_GREEN)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class HiDockApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HiDock P1")
        self.geometry("1150x900")
        self.minsize(950, 520)

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
        self.diarize_sessions = None  # (seg_sess, emb_sess) for diarization
        self.config = config.load()
        self.sort_key = "date"       # "date" or "duration"
        self.sort_desc = True        # descending by default

        self._build_ui()

        # Auto-connect to device on launch
        self.after(100, self._start_connect)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0,
                                     fg_color=CLR_GREEN_DARK)
        self.sidebar.grid(row=0, column=0, sticky="nswe")
        self.sidebar.grid_propagate(False)

        btn_font = ctk.CTkFont(size=14, weight="bold")
        btn_h = 38
        btn_pad = 5
        btn_radius = 10

        self.title_label = ctk.CTkLabel(
            self.sidebar, text="HiDock P1",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=CLR_TEXT_BRIGHT
        )
        self.title_label.pack(padx=20, pady=(20, 2))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, text="USB Recording Manager",
            font=ctk.CTkFont(size=13), text_color=CLR_TEXT_DIM
        )
        self.subtitle_label.pack(padx=20, pady=(0, 16))

        self.connect_btn = ctk.CTkButton(
            self.sidebar, text="Connect", command=self._on_connect,
            fg_color=CLR_GREEN, hover_color="#4ea02a", text_color=CLR_TEXT_BRIGHT,
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.connect_btn.pack(padx=20, pady=btn_pad, fill="x")

        # Device info area
        self.info_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.info_frame.pack(padx=20, pady=(10, 4), fill="x")

        info_font = ctk.CTkFont(size=12)
        for attr in ("fw_label", "serial_label"):
            lbl = ctk.CTkLabel(self.info_frame, text="", anchor="w",
                               font=info_font, text_color=CLR_TEXT)
            lbl.pack(fill="x")
            setattr(self, attr, lbl)

        # Battery row: icon canvas + text label
        self.battery_row = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        self.battery_row.pack(fill="x")
        self.battery_canvas = tk.Canvas(
            self.battery_row, width=28, height=14, bg=CLR_BG_PANEL,
            highlightthickness=0
        )
        self.battery_canvas.pack(side="left", pady=2)
        self.battery_label = ctk.CTkLabel(
            self.battery_row, text="", anchor="w",
            font=info_font, text_color=CLR_TEXT
        )
        self.battery_label.pack(side="left", padx=(4, 0))

        for attr in ("storage_label", "files_label"):
            lbl = ctk.CTkLabel(self.info_frame, text="", anchor="w",
                               font=info_font, text_color=CLR_TEXT)
            lbl.pack(fill="x")
            setattr(self, attr, lbl)

        sep = ctk.CTkFrame(self.sidebar, height=1, fg_color=CLR_SEP)
        sep.pack(padx=20, pady=10, fill="x")

        self.dl_selected_btn = ctk.CTkButton(
            self.sidebar, text="Download Selected",
            command=self._on_download_selected, state="disabled",
            fg_color="#89d466", hover_color="#4ea02a", text_color=CLR_GREEN_DARK,
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.dl_selected_btn.pack(padx=20, pady=btn_pad, fill="x")

        self.delete_btn = ctk.CTkButton(
            self.sidebar, text="Delete Selected",
            command=self._on_delete_selected, state="disabled",
            fg_color="#e6be6a", hover_color="#c07a10", text_color="#6b4a0d",
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.delete_btn.pack(padx=20, pady=btn_pad, fill="x")

        sep2 = ctk.CTkFrame(self.sidebar, height=1, fg_color=CLR_SEP)
        sep2.pack(padx=20, pady=10, fill="x")

        self.transcribe_btn = ctk.CTkButton(
            self.sidebar, text="Transcribe",
            command=self._on_transcribe, state="disabled",
            fg_color="#f5ee6e", hover_color=CLR_GOLD, text_color="#6b6b04",
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.transcribe_btn.pack(padx=20, pady=btn_pad, fill="x")

        # Diarization toggle
        self.diarize_var = ctk.BooleanVar(value=self.config.get("diarize_enabled", False))
        self.diarize_cb = ctk.CTkCheckBox(
            self.sidebar, text="Speaker Diarization",
            variable=self.diarize_var, command=self._on_diarize_toggle,
            font=ctk.CTkFont(size=12), text_color=CLR_TEXT,
            checkbox_width=20, checkbox_height=20
        )
        self.diarize_cb.pack(padx=24, pady=(2, 0), anchor="w")

        # Timecode toggle
        self.timecode_var = ctk.BooleanVar(value=self.config.get("show_timecodes", False))
        self.timecode_cb = ctk.CTkCheckBox(
            self.sidebar, text="Show Timecodes",
            variable=self.timecode_var, command=self._on_timecode_toggle,
            font=ctk.CTkFont(size=12), text_color=CLR_TEXT,
            checkbox_width=20, checkbox_height=20
        )
        self.timecode_cb.pack(padx=24, pady=(2, 4), anchor="w")

        # Model status labels
        model_font = ctk.CTkFont(size=11)
        self.diarize_model_label = ctk.CTkLabel(
            self.sidebar, text="Diarize: not loaded",
            font=model_font, text_color=CLR_TEXT_DIM
        )
        self.diarize_model_label.pack(padx=24, pady=(2, 0), anchor="w")

        self.model_label = ctk.CTkLabel(
            self.sidebar, text="Whisper: not loaded",
            font=model_font, text_color=CLR_TEXT_DIM
        )
        self.model_label.pack(padx=24, pady=(0, 8), anchor="w")

        sep3 = ctk.CTkFrame(self.sidebar, height=1, fg_color=CLR_SEP)
        sep3.pack(padx=20, pady=10, fill="x")

        self.dl_dir_btn = ctk.CTkButton(
            self.sidebar, text="Set Download Folder",
            command=self._on_choose_download_dir,
            fg_color=CLR_GREEN, hover_color="#4ea02a", text_color=CLR_TEXT_BRIGHT,
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.dl_dir_btn.pack(padx=20, pady=btn_pad, fill="x")

        self.dl_dir_label = ctk.CTkLabel(
            self.sidebar, text=self._format_config_dir("download_dir"),
            font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM, wraplength=210
        )
        self.dl_dir_label.pack(padx=20, pady=(2, 4))

        self.output_dir_btn = ctk.CTkButton(
            self.sidebar, text="Set Transcript Folder",
            command=self._on_choose_output_dir,
            fg_color=CLR_GREEN, hover_color="#4ea02a", text_color=CLR_TEXT_BRIGHT,
            font=btn_font, height=btn_h, corner_radius=btn_radius
        )
        self.output_dir_btn.pack(padx=20, pady=btn_pad, fill="x")

        self.output_dir_label = ctk.CTkLabel(
            self.sidebar, text=self._format_config_dir("transcript_output_dir"),
            font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM, wraplength=210
        )
        self.output_dir_label.pack(padx=20, pady=(2, 8))

        # --- Main panel ---
        self.main_panel = ctk.CTkFrame(self, corner_radius=0,
                                        fg_color=CLR_BG_PANEL)
        self.main_panel.grid(row=0, column=1, sticky="nswe")
        self.main_panel.grid_rowconfigure(0, weight=5)  # file list (~60%)
        # row 1 = progress (fixed)
        self.main_panel.grid_rowconfigure(2, weight=3)  # transcript (~40%)
        # row 3 = NPU chart (fixed height strip)
        self.main_panel.grid_columnconfigure(0, weight=1)

        # File list container: header + scrollable area stacked
        file_container = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        file_container.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)
        file_container.grid_rowconfigure(1, weight=1)
        file_container.grid_columnconfigure(0, weight=1)

        # File list scrollable area (create first to measure internal offset)
        self.file_list_frame = ctk.CTkScrollableFrame(
            file_container, fg_color="transparent"
        )
        self.file_list_frame.grid(row=1, column=0, sticky="nswe", padx=0, pady=0)
        self.file_list_frame.grid_columnconfigure(0, weight=1)

        # Header — padx matches scrollable frame's internal border + scrollbar
        # CTkScrollableFrame has ~3px left border; scrollbar ~14px on right
        hdr_pad_l = 3
        hdr_pad_r = 17
        header_frame = ctk.CTkFrame(file_container, height=30,
                                     fg_color=CLR_HEADER, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="nwe", padx=(hdr_pad_l, hdr_pad_r), pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)

        hdr_font = ctk.CTkFont(size=12, weight="bold")
        sort_font = ctk.CTkFont(size=11, weight="bold")

        # Select-all checkbox
        self.select_all_var = ctk.BooleanVar(value=False)
        self.select_all_cb = ctk.CTkCheckBox(
            header_frame, variable=self.select_all_var, text="", width=28,
            checkbox_width=20, checkbox_height=20,
            command=self._on_select_all
        )
        self.select_all_cb.grid(row=0, column=0, padx=(4, 0))

        ctk.CTkLabel(
            header_frame, text="Name", anchor="w", font=hdr_font,
            text_color=CLR_TEXT
        ).grid(row=0, column=1, padx=(4, 4), pady=4, sticky="w")

        ctk.CTkLabel(
            header_frame, text="Size", anchor="e",
            font=hdr_font, width=_COL_WIDTHS[2], text_color=CLR_TEXT
        ).grid(row=0, column=2, padx=4, pady=4)

        self.sort_dur_btn = ctk.CTkButton(
            header_frame, text="Length v", width=_COL_WIDTHS[3], height=22,
            font=sort_font, fg_color="transparent", text_color=CLR_YELLOW,
            hover_color=CLR_GREEN_DARK, anchor="e",
            command=lambda: self._toggle_sort("duration")
        )
        self.sort_dur_btn.grid(row=0, column=3, padx=4, pady=3)

        self.sort_date_btn = ctk.CTkButton(
            header_frame, text="Date v", width=_COL_WIDTHS[4], height=22,
            font=sort_font, fg_color="transparent", text_color=CLR_YELLOW,
            hover_color=CLR_GREEN_DARK, anchor="w",
            command=lambda: self._toggle_sort("date")
        )
        self.sort_date_btn.grid(row=0, column=4, padx=4, pady=3)

        ctk.CTkLabel(
            header_frame, text="Mode", anchor="w",
            font=hdr_font, width=_COL_WIDTHS[5], text_color=CLR_TEXT
        ).grid(row=0, column=5, padx=4, pady=4)

        ctk.CTkLabel(
            header_frame, text="DL", anchor="center",
            font=hdr_font, width=_COL_WIDTHS[6], text_color=CLR_TEXT
        ).grid(row=0, column=6, padx=(0, 8), pady=4)

        # Placeholder when no files
        self.placeholder_label = ctk.CTkLabel(
            self.file_list_frame,
            text="Connect a device to view recordings",
            text_color=CLR_TEXT_DIM, font=ctk.CTkFont(size=14)
        )
        self.placeholder_label.pack(pady=40)

        # Progress area
        self.progress_frame = ctk.CTkFrame(self.main_panel, height=50, fg_color="transparent")
        self.progress_frame.grid(row=1, column=0, sticky="we", padx=12, pady=4)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.progress_frame, text="Ready", anchor="w",
            font=ctk.CTkFont(size=12), text_color=CLR_TEXT
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame,
                                                 progress_color=CLR_GREEN)
        self.progress_bar.grid(row=1, column=0, sticky="we", pady=(2, 0))
        self.progress_bar.set(0)

        # Transcription output
        transcript_header = ctk.CTkFrame(self.main_panel, height=30,
                                          fg_color=CLR_HEADER, corner_radius=0)
        transcript_header.grid(row=2, column=0, sticky="nwe", padx=0, pady=0)
        transcript_header.grid_propagate(False)
        ctk.CTkLabel(
            transcript_header, text="  Transcription Output",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=CLR_TEXT
        ).pack(side="left", padx=8, pady=4)

        ctk.CTkButton(
            transcript_header, text="Clear", width=50, height=22,
            font=ctk.CTkFont(size=11), fg_color=CLR_GREEN_DARK,
            hover_color=CLR_AMBER, text_color=CLR_TEXT_DIM,
            corner_radius=6, command=self._clear_transcript
        ).pack(side="right", padx=8, pady=4)

        self.transcript_box = ctk.CTkTextbox(
            self.main_panel, state="disabled",
            font=ctk.CTkFont(family="Consolas", size=13),
            fg_color=CLR_GREEN_DEEP, text_color=CLR_TEXT
        )
        self.transcript_box.grid(row=2, column=0, sticky="nswe", padx=0, pady=(30, 0))

        # Processor utilization strip (compact fixed-height sparkline)
        proc_frame = ctk.CTkFrame(self.main_panel, height=60,
                                   fg_color=CLR_BG_DARK, corner_radius=0)
        proc_frame.grid(row=3, column=0, sticky="swe", padx=0, pady=0)
        proc_frame.grid_propagate(False)
        proc_frame.grid_columnconfigure(0, weight=1)
        proc_frame.grid_rowconfigure(1, weight=1)

        proc_label = ctk.CTkLabel(
            proc_frame, text="  Processor Utilization", anchor="w",
            font=ctk.CTkFont(size=10), text_color=CLR_TEXT_DIM,
            fg_color="transparent"
        )
        proc_label.grid(row=0, column=0, sticky="w", padx=4, pady=(2, 0))

        self.proc_chart = ProcessorChart(proc_frame, height=38)
        self.proc_chart.grid(row=1, column=0, sticky="nswe", padx=2, pady=(0, 2))

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

    def _clear_transcript(self):
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.configure(state="disabled")

    def _set_buttons_state(self, connected=False, has_downloads=False):
        def _apply():
            if connected:
                self.dl_selected_btn.configure(state="normal", fg_color=CLR_GREEN, text_color=CLR_TEXT_BRIGHT)
                self.delete_btn.configure(state="normal", fg_color=CLR_AMBER, text_color=CLR_TEXT_BRIGHT)
            else:
                self.dl_selected_btn.configure(state="disabled", fg_color="#89d466", text_color=CLR_GREEN_DARK)
                self.delete_btn.configure(state="disabled", fg_color="#e6be6a", text_color="#6b4a0d")
            if has_downloads:
                self.transcribe_btn.configure(state="normal", fg_color=CLR_YELLOW, text_color=CLR_GREEN_DARK)
            else:
                self.transcribe_btn.configure(state="disabled", fg_color="#f5ee6e", text_color="#6b6b04")
        self.after(0, _apply)

    def _draw_battery_icon(self, pct):
        """Draw a cell-phone-style battery icon on the battery canvas."""
        c = self.battery_canvas
        c.delete("all")
        w, h = 24, 12
        x0, y0 = 1, 1
        # Outer shell
        c.create_rectangle(x0, y0, x0 + w, y0 + h, outline=CLR_TEXT, width=1)
        # Positive terminal nub
        c.create_rectangle(x0 + w, y0 + 3, x0 + w + 3, y0 + h - 3,
                           fill=CLR_TEXT, outline="")
        # Fill level
        fill_w = max(0, int((w - 2) * min(pct, 100) / 100))
        if pct > 50:
            fill_color = CLR_GREEN
        elif pct > 20:
            fill_color = CLR_YELLOW
        else:
            fill_color = CLR_RED
        if fill_w > 0:
            c.create_rectangle(x0 + 1, y0 + 1, x0 + 1 + fill_w, y0 + h - 1,
                               fill=fill_color, outline="")

    def _populate_device_info(self):
        def _apply():
            if self.device_info:
                self.fw_label.configure(text=f"FW: {self.device_info['version']}")
                self.serial_label.configure(text=f"SN: {self.device_info['serial']}")
            if self.battery_info:
                pct = self.battery_info['battery']
                status = self.battery_info['status']
                self.battery_label.configure(text=f"{pct}% ({status})")
                self._draw_battery_icon(pct)
            self._update_storage_info()
        self.after(0, _apply)

    def _update_storage_info(self):
        """Update storage and file count labels from self.files."""
        TOTAL_BYTES = 16 * 1024 * 1024 * 1024  # 16 GB
        if self.files:
            used = sum(f["size"] for f in self.files)
            used_mb = used / (1024 * 1024)
            total_mb = TOTAL_BYTES / (1024 * 1024)
            pct = used / TOTAL_BYTES * 100
            self.storage_label.configure(
                text=f"Storage: {used_mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)"
            )
            self.files_label.configure(text=f"Files: {len(self.files)}")
        else:
            self.storage_label.configure(text="")
            self.files_label.configure(text="")

    def _on_select_all(self):
        val = self.select_all_var.get()
        for row in self.file_rows:
            row.selected.set(val)

    def _on_diarize_toggle(self):
        self.config["diarize_enabled"] = self.diarize_var.get()
        config.save(self.config)

    def _on_timecode_toggle(self):
        self.config["show_timecodes"] = self.timecode_var.get()
        config.save(self.config)

    def _populate_file_list(self):
        def _apply():
            # Clear existing rows
            for row in self.file_rows:
                row.destroy()
            self.file_rows.clear()
            self.placeholder_label.pack_forget()
            self.select_all_var.set(False)

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
        self._start_connect()

    def _start_connect(self):
        self.connect_btn.configure(state="disabled", text="Connecting...")
        self._set_status("Connecting to device...")
        self._set_progress_mode(indeterminate=True)
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            # Clean up any stale handle (e.g. device was power-cycled)
            if self.dev is not None:
                with self.usb_lock:
                    try:
                        hidock_p1.close_device(self.dev)
                    except Exception:
                        pass
                    self.dev = None

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
            self.dev = None
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
        self.storage_label.configure(text="")
        self.files_label.configure(text="")
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

    def _start_download(self, rows):
        out_dir = self.config.get("download_dir", "")
        if not out_dir:
            out_dir = filedialog.askdirectory(title="Choose download folder")
            if not out_dir:
                return
        # Disable buttons during download
        self.dl_selected_btn.configure(state="disabled")
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
    # Delete
    # -----------------------------------------------------------------------

    def _on_delete_selected(self):
        selected = [r for r in self.file_rows if r.selected.get()]
        if not selected:
            self._set_status("No files selected")
            return
        names = [r.file_info["name"] for r in selected]
        msg = f"Delete {len(names)} file(s) from device?\n\n" + "\n".join(names)
        from tkinter import messagebox
        if not messagebox.askyesno("Confirm Delete", msg, parent=self):
            return
        self.delete_btn.configure(state="disabled")
        self.dl_selected_btn.configure(state="disabled")
        self.connect_btn.configure(state="disabled")
        threading.Thread(
            target=self._delete_worker, args=(selected,), daemon=True
        ).start()

    def _delete_worker(self, rows):
        total = len(rows)
        deleted = 0
        for idx, row in enumerate(rows, 1):
            name = row.file_info["name"]
            self._set_status(f"Deleting {idx}/{total}: {name}")
            self._set_progress(idx / total)
            try:
                with self.usb_lock:
                    result = hidock_p1.delete_file(self.dev, name)
                if result == "success":
                    deleted += 1
                    self.files = [f for f in self.files if f["name"] != name]
                else:
                    self._set_status(f"Delete {name}: {result}")
            except Exception as e:
                self._set_status(f"Delete failed ({name}): {e}")

        self._set_progress(1.0)
        self._set_status(f"Deleted {deleted}/{total} file(s) from device")
        self._populate_file_list()
        self.after(0, self._update_storage_info)
        self.after(10, self._check_existing_downloads)
        self.after(0, lambda: self.connect_btn.configure(
            state="normal", text="Disconnect" if self.dev else "Connect"
        ))
        has_downloads = any(r.downloaded_path for r in self.file_rows)
        self.after(20, lambda: self._set_buttons_state(
            connected=self.dev is not None, has_downloads=has_downloads
        ))

    # -----------------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------------

    def _on_transcribe(self):
        selected = [r for r in self.file_rows
                    if r.selected.get() and r.downloaded_path is not None]
        if not selected:
            self._set_status("No selected files with downloads to transcribe")
            return
        downloaded = selected
        self.transcribe_btn.configure(state="disabled")
        threading.Thread(
            target=self._transcribe_worker, args=(downloaded,), daemon=True
        ).start()

    def _load_whisper(self):
        """Lazy-load ONNX sessions + tokenizer for NPU transcription."""
        self.whisper_loading = True
        self._set_progress_mode(indeterminate=True)

        self._set_status("Loading Whisper on NPU (QNN)...")
        self.after(0, lambda: self.model_label.configure(text="Whisper: loading..."))
        try:
            self.npu_sessions = transcribe_npu.load_sessions()
            self.tokenizer = transcribe_npu.load_tokenizer()
            self.after(0, lambda: self.model_label.configure(
                text="Whisper: Large-V3-Turbo (NPU)", text_color=CLR_GREEN
            ))
            return True
        except Exception as e:
            self._set_status(f"Failed to load Whisper: {e}")
            self.after(0, lambda: self.model_label.configure(
                text="Whisper: load error", text_color=CLR_RED
            ))
            return False

    def _transcribe_file(self, mp3_path, chunk_callback=None, step_callback=None):
        """Transcribe a single file on the NPU."""
        return transcribe_npu.transcribe(
            self.npu_sessions, self.tokenizer, mp3_path,
            chunk_callback=chunk_callback, step_callback=step_callback
        )

    def _load_diarize_models(self):
        """Lazy-load diarization ONNX models (CPU)."""
        if self.diarize_sessions is not None:
            return True
        self._set_status("Loading diarization models (CPU)...")
        self.after(0, lambda: self.diarize_model_label.configure(text="Diarize: loading..."))
        try:
            self.diarize_sessions = diarize.load_models()
            self.after(0, lambda: self.diarize_model_label.configure(
                text="Diarize: pyannote+wespeaker (CPU)", text_color=CLR_GREEN
            ))
            return True
        except Exception as e:
            self._set_status(f"Failed to load diarization models: {e}")
            self.after(0, lambda: self.diarize_model_label.configure(
                text="Diarize: load error", text_color=CLR_RED
            ))
            return False

    def _format_diarized(self, results, show_timecodes):
        """Format diarization results as labeled transcript text.

        Args:
            results: list of (speaker_id, text, start_s, end_s).
            show_timecodes: whether to include [MM:SS] prefixes.

        Returns:
            Formatted transcript string.
        """
        lines = []
        for speaker_id, text, start_s, end_s in results:
            label = f"[Speaker {speaker_id + 1}]"
            if show_timecodes:
                m, s = divmod(int(start_s), 60)
                tc = f"[{m:02d}:{s:02d}]"
                lines.append(f"{tc} {label} {text}")
            else:
                lines.append(f"{label} {text}")
        return "\n".join(lines)

    def _transcribe_worker(self, rows):
        # Lazy-load Whisper model on first use
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

        use_diarize = self.diarize_var.get()
        show_timecodes = self.timecode_var.get()

        # Lazy-load diarization models if needed
        if use_diarize:
            if not self._load_diarize_models():
                self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
                return

        self.after(0, self.proc_chart.clear)

        # Compute total audio duration for time-based progress
        total_audio_s = sum(row.file_info["duration"] for row in rows)
        files_done_s = 0.0
        total = len(rows)
        output_dir = self.config.get("transcript_output_dir", "")

        for idx, row in enumerate(rows, 1):
            mp3_path = row.downloaded_path
            filename = os.path.basename(mp3_path)

            try:
                if use_diarize:
                    text = self._transcribe_file_diarized(
                        mp3_path, filename, idx, total,
                        files_done_s, total_audio_s, show_timecodes
                    )
                else:
                    text = self._transcribe_file_plain(
                        mp3_path, filename, idx, total,
                        files_done_s, total_audio_s
                    )

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

            files_done_s += row.file_info["duration"]

        self._set_progress(1.0)
        self.after(0, self.proc_chart.clear)
        saved_msg = f" — saved to {output_dir}" if output_dir else ""
        mode = "NPU+CPU" if use_diarize else "NPU"
        self._set_status(f"Transcription complete — {total} file(s) [{mode}]{saved_msg}")
        self.after(0, lambda: self.transcribe_btn.configure(state="normal"))

    def _transcribe_file_plain(self, mp3_path, filename, idx, total,
                               files_done_s, total_audio_s):
        """Transcribe a single file without diarization (original pipeline)."""
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

        def on_step(step_info):
            ms = step_info["run_time_ms"]
            is_enc = step_info["phase"] == "encoder"
            self.after(0, lambda: self.proc_chart.add_sample(ms, "npu", is_enc))

        return self._transcribe_file(
            mp3_path, chunk_callback=on_chunk, step_callback=on_step
        )

    def _transcribe_file_diarized(self, mp3_path, filename, idx, total,
                                  files_done_s, total_audio_s, show_timecodes):
        """Transcribe a single file with speaker diarization."""
        seg_sess, emb_sess = self.diarize_sessions

        # Load audio once for both diarization and transcription
        self._set_status(f"Loading audio {idx}/{total}: {filename}...")
        audio = transcribe_npu.load_audio_16k(mp3_path)
        file_duration = len(audio) / transcribe_npu.SAMPLE_RATE

        # Phase 1: Diarization (CPU)
        def on_diarize_step(info):
            ms = info["run_time_ms"]
            self.after(0, lambda: self.proc_chart.add_sample(ms, "cpu"))
            phase = info.get("phase", "")
            if phase == "segmentation":
                win = info.get("window", 0)
                total_win = info.get("total_windows", 0)
                self._set_status(
                    f"Diarizing {idx}/{total}: {filename} — "
                    f"segmentation {win}/{total_win}"
                )
            elif phase == "embedding":
                seg = info.get("segment", 0)
                total_seg = info.get("total_segments", 0)
                self._set_status(
                    f"Diarizing {idx}/{total}: {filename} — "
                    f"embedding {seg}/{total_seg}"
                )

        self._set_status(f"Diarizing {idx}/{total}: {filename}...")
        segments = diarize.diarize(
            seg_sess, emb_sess, audio, step_callback=on_diarize_step
        )
        num_speakers = len(set(s[2] for s in segments))
        num_segs = len(segments)

        # Phase 2: Transcribe per segment (NPU)
        def on_npu_step(step_info):
            ms = step_info["run_time_ms"]
            is_enc = step_info["phase"] == "encoder"
            self.after(0, lambda: self.proc_chart.add_sample(ms, "npu", is_enc))

        def on_segment(info):
            seg_idx = info["segment_index"]
            spk = info["speaker_id"] + 1
            pct_base = files_done_s / total_audio_s if total_audio_s > 0 else 0
            pct_file = (seg_idx + 1) / num_segs * (file_duration / total_audio_s) if total_audio_s > 0 else 0
            self._set_progress(pct_base + pct_file)
            self._set_status(
                f"Transcribing {idx}/{total}: {filename} — "
                f"segment {seg_idx + 1}/{num_segs} (Speaker {spk})"
            )

        results = transcribe_npu.transcribe_segments(
            self.npu_sessions, self.tokenizer, audio, segments,
            step_callback=on_npu_step, segment_callback=on_segment,
        )

        return self._format_diarized(results, show_timecodes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = HiDockApp()
    app.mainloop()
