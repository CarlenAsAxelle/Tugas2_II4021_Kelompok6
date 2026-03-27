# GUI.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.stego_video     import embed_message as _embed_avi, extract_message as _extract_avi
from src.stego_video_mp4 import embed_message as _embed_mp4, extract_message as _extract_mp4
from src.video_io        import read_video_frames as _read_avi, color_histogram_video, mse_psnr_video
from src.video_io_mp4    import read_video_frames as _read_mp4
from src.stego_lsb       import capacity_332

def _read_video(path):
    """Read video frames, routing to correct backend by extension."""
    if _fmt_of(path) == 'mp4':
        return _read_mp4(path)
    else:
        return _read_avi(path)

def _fmt_of(path):
    """Return lowercase extension without dot: 'avi' or 'mp4'."""
    return os.path.splitext(path)[1].lower().lstrip('.')

# ─── PALETTE ──────────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
PANEL   = "#16213e"
CARD    = "#0f3460"
ACCENT  = "#e94560"
ACCENT2 = "#533483"
TEXT    = "#eaeaea"
MUTED   = "#8892a4"
SUCCESS = "#4ecca3"
WARNING = "#f5a623"
DANGER  = "#ff4757"
BTN_BG  = "#0f3460"
BTN_FG  = "#eaeaea"
BTN_ACT = "#e94560"

DISPLAY_W = 420
DISPLAY_H = 240


def _resize_for_display(frame, w=DISPLAY_W, h=DISPLAY_H):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

def _make_btn(parent, text, command, bg=BTN_BG, fg=BTN_FG, **kw):
    return tk.Button(parent, text=text, command=command,
                     bg=bg, fg=fg, activebackground=BTN_ACT,
                     activeforeground="white", relief="flat",
                     padx=10, pady=4, cursor="hand2", **kw)

def _make_label(parent, text="", fg=TEXT, font=None, **kw):
    return tk.Label(parent, text=text,
                    bg=kw.pop("bg", PANEL),
                    fg=fg, font=font or ("Segoe UI", 9), **kw)

def _fmt_bytes(n):
    if n < 1024:        return f"{n} B"
    if n < 1024**2:     return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


class StegoApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography Dashboard")
        self.root.geometry("1280x800")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.video_frames      = []
        self.fps               = 30
        self.current_frame     = 0
        self.playing           = False
        self.video_selected    = False
        self.video_path        = None
        self.video_format      = None   # 'avi' or 'mp4'
        self._play_job         = None
        self._capacity_bytes   = 0
        self._embed_file_path  = None
        self._embed_file_bytes = 0

        self.cmp_cover_frames = []
        self.cmp_stego_frames = []
        self.cmp_cover_path   = None
        self.cmp_stego_path   = None
        self.cmp_idx          = 0
        self.cmp_mse_list     = []
        self.cmp_psnr_list    = []

        self._apply_style()
        self._setup_ui()

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",     background=BG,   borderwidth=0)
        s.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                    padding=[16, 6], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT2)],
              foreground=[("selected", TEXT)])
        s.configure("TScale",            background=BG, troughcolor=PANEL)
        s.configure("Horizontal.TScale", background=BG)

    def _setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.embed_tab   = tk.Frame(self.notebook, bg=BG)
        self.extract_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.embed_tab,   text="  📥  Embed  ")
        self.notebook.add(self.extract_tab, text="  📤  Extract  ")
        self._build_embed_ui()
        self._build_extract_ui()

    # ══════════════════════════════════════════════════════════════════════════
    #  EMBED TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_embed_ui(self):
        root = self.embed_tab
        root.columnconfigure(0, weight=1, minsize=340)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(2, weight=1)

        left = tk.Frame(root, bg=PANEL, padx=12, pady=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(6,3), pady=6)
        left.columnconfigure(0, weight=1)
        r = 0

        _make_label(left, "VIDEO SOURCE", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w", pady=(0,3)); r+=1
        self.embed_select_btn = _make_btn(left, "📂  Select Cover Video", self._toggle_video, bg=CARD)
        self.embed_select_btn.grid(row=r, column=0, sticky="ew"); r+=1
        self.embed_video_lbl = _make_label(left, "No video selected", fg=MUTED, bg=PANEL, font=("Segoe UI",8))
        self.embed_video_lbl.grid(row=r, column=0, sticky="w", pady=(2,0)); r+=1

        # ── capacity bar
        cap_frame = tk.Frame(left, bg=CARD, padx=8, pady=6)
        cap_frame.grid(row=r, column=0, sticky="ew", pady=(6,0)); r+=1
        cap_frame.columnconfigure(0, weight=1)
        _make_label(cap_frame, "CAPACITY", fg=ACCENT, font=("Segoe UI",7,"bold"), bg=CARD).grid(row=0, column=0, sticky="w")
        self.cap_total_lbl = _make_label(cap_frame, "Load a video first", fg=MUTED, bg=CARD, font=("Consolas",8))
        self.cap_total_lbl.grid(row=1, column=0, sticky="w")
        self.cap_used_lbl = _make_label(cap_frame, "", fg=TEXT, bg=CARD, font=("Consolas",8))
        self.cap_used_lbl.grid(row=2, column=0, sticky="w")
        self.cap_canvas = tk.Canvas(cap_frame, height=8, bg=PANEL, highlightthickness=0)
        self.cap_canvas.grid(row=3, column=0, sticky="ew", pady=(4,0))

        tk.Frame(left, bg=ACCENT2, height=1).grid(row=r, column=0, sticky="ew", pady=8); r+=1

        # ── message
        _make_label(left, "MESSAGE (text)", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w"); r+=1
        _make_label(left, "Type here — or leave blank and select a file below", fg=MUTED, font=("Segoe UI",7), bg=PANEL).grid(row=r, column=0, sticky="w", pady=(0,3)); r+=1
        self.message_entry = tk.Text(left, height=4, bg=CARD, fg=TEXT, insertbackground=TEXT, font=("Consolas",9), relief="flat", padx=6, pady=4)
        self.message_entry.grid(row=r, column=0, sticky="ew"); r+=1
        self.message_entry.bind("<KeyRelease>", self._on_message_change)

        file_row = tk.Frame(left, bg=PANEL)
        file_row.grid(row=r, column=0, sticky="ew", pady=(6,0)); r+=1
        file_row.columnconfigure(1, weight=1)
        _make_btn(file_row, "📁  Select File to Embed", self._select_file_to_embed, bg=CARD, font=("Segoe UI",8)).grid(row=0, column=0, sticky="w")
        self.embed_file_lbl = _make_label(file_row, "No file selected", fg=MUTED, bg=PANEL, font=("Segoe UI",7))
        self.embed_file_lbl.grid(row=0, column=1, sticky="w", padx=(6,0))

        tk.Frame(left, bg=ACCENT2, height=1).grid(row=r, column=0, sticky="ew", pady=8); r+=1

        # ── encryption
        _make_label(left, "ENCRYPTION (A5/1)", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w"); r+=1
        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(left, text="Enable A5/1 Encryption", variable=self.encrypt_var, bg=PANEL, fg=TEXT, selectcolor=CARD, activebackground=PANEL, font=("Segoe UI",9)).grid(row=r, column=0, sticky="w"); r+=1
        self.key_entry = tk.Entry(left, bg=CARD, fg=MUTED, insertbackground=TEXT, relief="flat", font=("Consolas",9))
        self.key_entry.insert(0, "e.g. 0x123456789ABCDEF0")
        self.key_entry.grid(row=r, column=0, sticky="ew", pady=(2,6)); r+=1
        self._bind_placeholder(self.key_entry, "e.g. 0x123456789ABCDEF0")

        # ── random
        _make_label(left, "RANDOM EMBEDDING", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w"); r+=1
        self.random_var = tk.BooleanVar()
        tk.Checkbutton(left, text="Enable Random Pixel Order", variable=self.random_var, bg=PANEL, fg=TEXT, selectcolor=CARD, activebackground=PANEL, font=("Segoe UI",9)).grid(row=r, column=0, sticky="w"); r+=1
        self.stego_key_entry = tk.Entry(left, bg=CARD, fg=MUTED, insertbackground=TEXT, relief="flat", font=("Consolas",9))
        self.stego_key_entry.insert(0, "Stego key (integer)")
        self.stego_key_entry.grid(row=r, column=0, sticky="ew", pady=(2,8)); r+=1
        self._bind_placeholder(self.stego_key_entry, "Stego key (integer)")

        self.embed_btn = _make_btn(left, "🔒  Embed Message", self._run_embed, bg=ACCENT, fg="white", font=("Segoe UI",10,"bold"))
        self.embed_btn.grid(row=r, column=0, sticky="ew", ipady=4); r+=1
        self.embed_status = _make_label(left, "", fg=MUTED, font=("Segoe UI",8), bg=PANEL)
        self.embed_status.grid(row=r, column=0, sticky="w", pady=(4,0)); r+=1
        self.key_saved_label = _make_label(left, "", fg=WARNING, font=("Segoe UI",8), bg=PANEL)
        self.key_saved_label.grid(row=r, column=0, sticky="w", pady=(1,0))

        # ── right: preview
        right = tk.Frame(root, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew", padx=(3,6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        _make_label(right, "VIDEO PREVIEW", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=0, column=0, pady=(8,4))
        self.embed_canvas = tk.Label(right, bg="#000", width=DISPLAY_W, height=DISPLAY_H)
        self.embed_canvas.grid(row=1, column=0, padx=8)
        ctrl = tk.Frame(right, bg=PANEL)
        ctrl.grid(row=2, column=0, pady=6)
        _make_btn(ctrl, "▶  Play",  self._play_video,  bg=CARD).pack(side="left", padx=4)
        _make_btn(ctrl, "⏸  Pause", self._pause_video, bg=CARD).pack(side="left", padx=4)

        metrics = tk.Frame(root, bg=CARD, pady=5)
        metrics.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6)
        self.metrics_label = _make_label(metrics, "MSE: —  |  PSNR: —", fg=SUCCESS, font=("Consolas",10), bg=CARD)
        self.metrics_label.pack()

        self.chart_frame = tk.Frame(root, bg=BG)
        self.chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)

    # ── capacity bar ─────────────────────────────────────────────────────────
    def _update_capacity_ui(self, msg_size_bytes=0):
        cap    = self._capacity_bytes
        HEADER = 32
        if cap == 0:
            self.cap_total_lbl.config(text="Load a video first", fg=MUTED)
            self.cap_used_lbl.config(text="")
            self.cap_canvas.delete("all")
            return
        needed = HEADER + msg_size_bytes
        pct    = min(needed / cap, 1.0)
        self.cap_total_lbl.config(text=f"Total capacity: {_fmt_bytes(cap)}  (header: {HEADER} B)", fg=MUTED)
        if msg_size_bytes == 0:
            self.cap_used_lbl.config(text="Select a message to see usage", fg=MUTED)
        else:
            fits  = needed <= cap
            color = SUCCESS if fits else DANGER
            self.cap_used_lbl.config(
                text=(f"Message: {_fmt_bytes(msg_size_bytes)}  +  Header: {_fmt_bytes(HEADER)}  "
                      f"=  {_fmt_bytes(needed)}  "
                      f"({'✓ fits' if fits else '✗ TOO LARGE'})"),
                fg=color)
        self.cap_canvas.update_idletasks()
        W = self.cap_canvas.winfo_width() or 260
        fill = SUCCESS if pct < 0.9 else (WARNING if pct < 1.0 else DANGER)
        self.cap_canvas.delete("all")
        self.cap_canvas.create_rectangle(0, 0, W, 8, fill=PANEL, outline="")
        self.cap_canvas.create_rectangle(0, 0, int(W * pct), 8, fill=fill, outline="")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _bind_placeholder(self, entry, placeholder):
        def on_in(e):
            if entry.get() == placeholder:
                entry.delete(0, tk.END); entry.config(fg=TEXT)
        def on_out(e):
            if not entry.get():
                entry.insert(0, placeholder); entry.config(fg=MUTED)
        entry.bind("<FocusIn>",  on_in)
        entry.bind("<FocusOut>", on_out)

    def _get_key(self, entry, placeholder):
        v = entry.get().strip()
        return None if (not v or v == placeholder) else v

    def _on_message_change(self, event=None):
        text = self.message_entry.get("1.0", tk.END).strip()
        if text:
            self._embed_file_path  = None
            self._embed_file_bytes = 0
            self.embed_file_lbl.config(text="No file selected", fg=MUTED)
            self._update_capacity_ui(len(text.encode("utf-8")))
        elif self._embed_file_bytes:
            self._update_capacity_ui(self._embed_file_bytes)
        else:
            self._update_capacity_ui(0)

    def _select_file_to_embed(self):
        path = filedialog.askopenfilename(title="Select file to embed")
        if not path:
            return
        size = os.path.getsize(path)
        self._embed_file_path  = path
        self._embed_file_bytes = size
        self.embed_file_lbl.config(text=f"{os.path.basename(path)}  ({_fmt_bytes(size)})", fg=TEXT)
        self.message_entry.delete("1.0", tk.END)
        self._update_capacity_ui(size)

    def _toggle_video(self):
        if not self.video_selected:
            path = filedialog.askopenfilename(
                title="Select Cover Video",
                filetypes=[("Video files", "*.avi *.mp4")])
            if not path:
                return
            self.video_path = path
            self.embed_video_lbl.config(text=os.path.basename(path), fg=TEXT)
            self.embed_status.config(text="Loading video…", fg=WARNING)
            self.embed_select_btn.config(state="disabled")
            threading.Thread(target=self._load_video_worker, args=(path,), daemon=True).start()
        else:
            if self._play_job:
                self.root.after_cancel(self._play_job); self._play_job = None
            self.playing = False; self.video_path = None; self.video_frames = []
            self.video_selected = False; self._capacity_bytes = 0; self.video_format = None
            self.embed_canvas.config(image="")
            self.embed_video_lbl.config(text="No video selected", fg=MUTED)
            self.embed_select_btn.config(text="📂  Select Cover Video", state="normal")
            self.embed_status.config(text="", fg=MUTED)
            self._update_capacity_ui(0)

    def _load_video_worker(self, path):
        try:
            frames, fps = _read_video(path)
            cap = sum(capacity_332(f) for f in frames) // 8
            fmt = _fmt_of(path)
            self.root.after(0, self._load_video_done, frames, fps, cap, fmt)
        except Exception as exc:
            self.root.after(0, self._load_video_error, str(exc))

    def _load_video_done(self, frames, fps, cap, fmt):
        self.video_frames = frames; self.fps = fps; self._capacity_bytes = cap
        self.video_format = fmt
        self.video_selected = True
        self.embed_select_btn.config(text="✖  Clear Video", state="normal")
        self.embed_status.config(
            text=f"✅  Loaded {len(frames)} frames  [{fmt.upper()}]", fg=SUCCESS)
        self._update_capacity_ui(0)
        self._play_video()

    def _load_video_error(self, msg):
        self.embed_select_btn.config(state="normal")
        self.embed_status.config(text=f"❌  {msg}", fg=DANGER)

    def _play_video(self):
        self.playing = True; self._update_frame()

    def _pause_video(self):
        self.playing = False

    def _update_frame(self):
        if not self.playing or not self.video_frames:
            return
        frame = self.video_frames[self.current_frame]
        img   = _resize_for_display(frame)
        self.embed_canvas.configure(image=img); self.embed_canvas.image = img
        self.current_frame = (self.current_frame + 1) % len(self.video_frames)
        self._play_job = self.root.after(int(1000/max(self.fps,1)), self._update_frame)

    def _run_embed(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a cover video first."); return

        # Collect payload
        text = self.message_entry.get("1.0", tk.END).strip()
        if text:
            msg = text.encode("utf-8"); is_text = True
            extension = ".txt"; filename = "message.txt"
        elif self._embed_file_path:
            with open(self._embed_file_path, "rb") as f:
                msg = f.read()
            is_text = False
            extension = os.path.splitext(self._embed_file_path)[1]
            filename  = os.path.basename(self._embed_file_path)
        else:
            messagebox.showerror("Error", "Enter a text message or select a file to embed."); return

        # ── Capacity pre-check (BEFORE spawning thread) ───────────────────────
        HEADER   = 32
        needed   = HEADER + len(msg)
        capacity = self._capacity_bytes
        if capacity > 0 and needed > capacity:
            messagebox.showerror(
                "Message Too Large",
                f"Cannot embed — message exceeds video capacity.\n\n"
                f"  Message size   : {_fmt_bytes(len(msg))}\n"
                f"  Header         : {_fmt_bytes(HEADER)}\n"
                f"  Total needed   : {_fmt_bytes(needed)}\n"
                f"  Video capacity : {_fmt_bytes(capacity)}\n\n"
                f"Use a shorter message or a larger cover video.")
            return

        # Parse keys
        use_encrypt = self.encrypt_var.get(); use_random = self.random_var.get()
        a51_key = None; stego_key = None

        if use_encrypt:
            raw = self._get_key(self.key_entry, "e.g. 0x123456789ABCDEF0")
            if not raw:
                import random; a51_key = random.getrandbits(64)
                messagebox.showinfo("Auto-generated Key",
                    f"Generated A5/1 key:\n0x{a51_key:016x}\n\nThis will be saved to the keys file.")
            else:
                try: a51_key = int(raw, 0)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid A5/1 key: '{raw}'"); return

        if use_random:
            raw = self._get_key(self.stego_key_entry, "Stego key (integer)")
            if not raw:
                import random; stego_key = random.getrandbits(32)
                messagebox.showinfo("Auto-generated Key",
                    f"Generated stego key:\n{stego_key}\n\nThis will be saved to the keys file.")
            else:
                try: stego_key = int(raw)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid stego key: '{raw}'"); return

        # Output path — locked to same format as the input video
        fmt = self.video_format or 'avi'
        if fmt == 'mp4':
            out_ext   = ".mp4"
            out_types = [("MP4 video", "*.mp4")]
        else:
            out_ext   = ".avi"
            out_types = [("AVI video", "*.avi")]

        output_path = filedialog.asksaveasfilename(
            title=f"Save stego video as ({fmt.upper()})",
            defaultextension=out_ext,
            filetypes=out_types)
        if not output_path:
            return

        self.embed_btn.config(state="disabled", text="⏳  Embedding…")
        self.embed_status.config(text="Processing… please wait.", fg=WARNING)
        self.key_saved_label.config(text="")
        self.metrics_label.config(text="MSE: —  |  PSNR: —")

        threading.Thread(
            target=self._embed_worker,
            args=(msg, is_text, extension, filename,
                  use_encrypt, a51_key, use_random, stego_key, output_path),
            daemon=True
        ).start()

    def _embed_worker(self, msg, is_text, extension, filename,
                      use_encrypt, a51_key, use_random, stego_key, output_path):
        try:
            # Route to the correct backend based on output file format
            embed_fn = _embed_mp4 if _fmt_of(output_path) == 'mp4' else _embed_avi
            result = embed_fn(
                cover_path=self.video_path, output_path=output_path,
                message=msg, is_text=is_text, extension=extension, filename=filename,
                use_encryption=use_encrypt, a51_key=a51_key,
                use_random=use_random, stego_key=stego_key)
            cover_frames    = self.video_frames
            stego_frames, _ = _read_video(output_path)
            key_path = None
            if use_encrypt or use_random:
                key_path = self._save_key_file(output_path, a51_key, stego_key)
            self.root.after(0, self._embed_done, result, cover_frames, stego_frames, key_path)
        except Exception as exc:
            self.root.after(0, self._embed_error, str(exc))

    def _embed_done(self, result, cover_frames, stego_frames, key_path):
        self.embed_btn.config(state="normal", text="🔒  Embed Message")
        self.embed_status.config(text="✅  Embedding complete!", fg=SUCCESS)
        self.metrics_label.config(
            text=(f"MSE: {result['mse_avg']:.4f}  |  "
                  f"PSNR: {result['psnr_avg']:.2f} dB  |  "
                  f"Capacity: {_fmt_bytes(result['total_capacity_bytes'])}"),
            fg=SUCCESS)
        if key_path:
            self.key_saved_label.config(text=f"🔑  Keys saved → {key_path}", fg=WARNING)
            messagebox.showinfo("Keys Saved",
                f"Keys saved to:\n\n{os.path.abspath(key_path)}\n\n"
                "Keep this file safe — you need it to extract the message.")
        self._show_histogram(cover_frames, stego_frames, result["mse_list"])

    def _embed_error(self, msg):
        self.embed_btn.config(state="normal", text="🔒  Embed Message")
        self.embed_status.config(text=f"❌  {msg}", fg=DANGER)
        messagebox.showerror("Embed Error", msg)

    def _save_key_file(self, output_path, a51_key, stego_key):
        base = os.path.splitext(output_path)[0]
        key_path = base + "_keys.txt"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = ["=== Video Steganography — Key File ===",
                 f"Generated : {ts}", f"Video     : {os.path.basename(output_path)}", ""]
        if a51_key  is not None: lines.append(f"A5/1 Key  : {hex(a51_key)}")
        if stego_key is not None: lines.append(f"Stego Key : {stego_key}")
        lines += ["", "KEEP THIS FILE SAFE.",
                  "You will need these keys to extract the hidden message."]
        with open(key_path, "w") as fh:
            fh.write("\n".join(lines))
        return key_path

    def _show_histogram(self, cover_frames, stego_frames, mse_list):
        hb_c, hg_c, hr_c = color_histogram_video(cover_frames)
        hb_s, hg_s, hr_s = color_histogram_video(stego_frames)
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.2), facecolor="#0f3460")
        fig.suptitle(f"Color Histogram — Cover vs Stego  (MSE avg={np.mean(mse_list):.4f})",
                     fontsize=9, color=TEXT)
        for ax, (hc, hs, ch) in zip(axes,
                [(hb_c,hb_s,"#4e9af1"),(hg_c,hg_s,"#4ecca3"),(hr_c,hr_s,"#e94560")]):
            ax.plot(hc, color=ch, alpha=0.8, linewidth=1, label="Cover")
            ax.plot(hs, color=ch, alpha=0.5, linewidth=1, linestyle="--", label="Stego")
            ax.set_facecolor("#16213e")
            ax.tick_params(colors=MUTED, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(CARD)
        axes[0].set_title("Blue",  color=TEXT, fontsize=8)
        axes[1].set_title("Green", color=TEXT, fontsize=8)
        axes[2].set_title("Red",   color=TEXT, fontsize=8)
        axes[1].legend(fontsize=6, facecolor=CARD, labelcolor=TEXT)
        fig.tight_layout()
        for w in self.chart_frame.winfo_children(): w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXTRACT TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_extract_ui(self):
        root = self.extract_tab
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        left = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(6,3), pady=6)
        left.columnconfigure(0, weight=1)
        r = 0

        _make_label(left, "STEGO VIDEO", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w", pady=(0,4)); r+=1
        _make_btn(left, "📂  Select Stego Video", self._select_stego, bg=CARD).grid(row=r, column=0, sticky="ew"); r+=1
        self.stego_lbl = _make_label(left, "No file selected", fg=MUTED, font=("Segoe UI",8), bg=PANEL)
        self.stego_lbl.grid(row=r, column=0, sticky="w", pady=(2,12)); r+=1

        _make_label(left, "A5/1 KEY", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w"); r+=1
        _make_label(left, "Leave blank if encryption was NOT used", fg=MUTED, font=("Segoe UI",7), bg=PANEL).grid(row=r, column=0, sticky="w", pady=(0,2)); r+=1
        self.extract_key = tk.Entry(left, bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat", font=("Consolas",9))
        self.extract_key.grid(row=r, column=0, sticky="ew", pady=(0,10)); r+=1

        _make_label(left, "STEGO KEY", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=r, column=0, sticky="w"); r+=1
        _make_label(left, "Leave blank if random mode was NOT used", fg=MUTED, font=("Segoe UI",7), bg=PANEL).grid(row=r, column=0, sticky="w", pady=(0,2)); r+=1
        self.extract_stego_key = tk.Entry(left, bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat", font=("Consolas",9))
        self.extract_stego_key.grid(row=r, column=0, sticky="ew", pady=(0,14)); r+=1

        self.extract_btn = _make_btn(left, "🔓  Extract Message", self._run_extract, bg=ACCENT, fg="white", font=("Segoe UI",10,"bold"))
        self.extract_btn.grid(row=r, column=0, sticky="ew", ipady=4); r+=1
        self.extract_status = _make_label(left, "", fg=MUTED, font=("Segoe UI",8), bg=PANEL)
        self.extract_status.grid(row=r, column=0, sticky="w", pady=(4,0)); r+=1
        self.extract_file_lbl = _make_label(left, "", fg=WARNING, font=("Segoe UI",8), bg=PANEL)
        self.extract_file_lbl.grid(row=r, column=0, sticky="w", pady=(2,0))

        # right panel
        right = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(3,6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        hdr = tk.Frame(right, bg=PANEL)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,4))
        hdr.columnconfigure(0, weight=1)
        _make_label(hdr, "EXTRACTED MESSAGE", fg=ACCENT, font=("Segoe UI",8,"bold"), bg=PANEL).grid(row=0, column=0, sticky="w")
        self.copy_btn = _make_btn(hdr, "📋  Copy", self._copy_extracted, bg=CARD, font=("Segoe UI",8))
        self.copy_btn.grid(row=0, column=1, sticky="e")

        self.extract_meta_lbl = _make_label(right, "", fg=MUTED, font=("Consolas",8), bg=PANEL)
        self.extract_meta_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0,4))

        self.output_text = tk.Text(right, bg=CARD, fg=SUCCESS, insertbackground=TEXT, relief="flat",
                                   font=("Consolas",10), padx=8, pady=8, wrap="word", state="disabled")
        self.output_text.grid(row=2, column=0, sticky="nsew")
        sb = tk.Scrollbar(right, command=self.output_text.yview, bg=PANEL)
        sb.grid(row=2, column=1, sticky="ns")
        self.output_text.config(yscrollcommand=sb.set)

    def _select_stego(self):
        path = filedialog.askopenfilename(title="Select Stego Video",
                                          filetypes=[("Video files", "*.avi *.mp4")])
        if path:
            self.stego_path = path
            self.stego_lbl.config(text=os.path.basename(path), fg=TEXT)

    def _set_output_text(self, text, color=SUCCESS):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled", fg=color)

    def _copy_extracted(self):
        content = self.output_text.get("1.0", tk.END).strip()
        if content:
            self.root.clipboard_clear(); self.root.clipboard_append(content)
            self.copy_btn.config(text="✅  Copied!")
            self.root.after(2000, lambda: self.copy_btn.config(text="📋  Copy"))

    def _run_extract(self):
        if not hasattr(self, "stego_path") or not self.stego_path:
            messagebox.showerror("Error", "Please select a stego video first."); return
        raw_key = self.extract_key.get().strip()
        raw_sk  = self.extract_stego_key.get().strip()
        a51_key = None; stego_key = None
        if raw_key:
            try: a51_key = int(raw_key, 0)
            except ValueError:
                messagebox.showerror("Error", f"Invalid A5/1 key: '{raw_key}'"); return
        if raw_sk:
            try: stego_key = int(raw_sk)
            except ValueError:
                messagebox.showerror("Error", f"Invalid stego key: '{raw_sk}'"); return

        self.extract_btn.config(state="disabled", text="⏳  Extracting…")
        self.extract_status.config(text="Processing… please wait.", fg=WARNING)
        self.extract_file_lbl.config(text="")
        self.extract_meta_lbl.config(text="")
        self._set_output_text("⏳  Extracting message…", WARNING)
        threading.Thread(target=self._extract_worker,
                         args=(self.stego_path, a51_key, stego_key), daemon=True).start()

    def _extract_worker(self, stego_path, a51_key, stego_key):
        try:
            # Route to the correct backend based on stego file format
            extract_fn = _extract_mp4 if _fmt_of(stego_path) == 'mp4' else _extract_avi
            result = extract_fn(stego_path, a51_key, stego_key)
            self.root.after(0, self._extract_done, result, stego_path)
        except Exception as exc:
            self.root.after(0, self._extract_error, str(exc))

    def _extract_done(self, result, stego_path):
        self.extract_btn.config(state="normal", text="🔓  Extract Message")
        self.extract_status.config(text="✅  Extraction complete!", fg=SUCCESS)

        mode_str = "Random" if result["is_random"] else "Sequential"
        enc_str  = "Yes"    if result["is_encrypted"] else "No"
        type_str = "Text"   if result["is_text"] else f"File ({result['extension'] or '.bin'})"
        self.extract_meta_lbl.config(
            text=(f"Type: {type_str}   |   Mode: {mode_str}   |   "
                  f"Encrypted: {enc_str}   |   Size: {_fmt_bytes(result['payload_size'])}"),
            fg=MUTED)

        if result["is_text"]:
            # ── TEXT: display directly in the UI ─────────────────────────────
            try:
                text = result["message"].decode("utf-8")
            except UnicodeDecodeError:
                text = result["message"].decode("latin-1", errors="replace")
            self._set_output_text(text, SUCCESS)
            self.extract_file_lbl.config(text="", fg=WARNING)

        else:
            # ── FILE: Save As dialog with original filename as default ─────────
            ext       = result["extension"] or ".bin"
            orig_name = result.get("filename", "") or f"extracted{ext}"
            if not orig_name.lower().endswith(ext.lower()):
                orig_name = os.path.splitext(orig_name)[0] + ext

            self._set_output_text(
                f"Binary file ready to save.\n\n"
                f"Original filename : {orig_name}\n"
                f"File type         : {ext}\n"
                f"Size              : {_fmt_bytes(result['payload_size'])}\n\n"
                f"A Save As dialog will open — choose where to save it.",
                WARNING)

            # Open Save As after render (slight delay so text appears first)
            self.root.after(150, lambda: self._prompt_save_file(
                result["message"], orig_name, ext))

    def _prompt_save_file(self, data, default_name, ext):
        """Save As dialog with original filename as default value."""
        save_path = filedialog.asksaveasfilename(
            title="Save extracted file",
            initialfile=default_name,
            defaultextension=ext,
            filetypes=[(f"{ext.upper()} file", f"*{ext}"), ("All files", "*.*")])
        if not save_path:
            self.extract_file_lbl.config(
                text="⚠  Save cancelled. Call Extract again to re-save.", fg=WARNING)
            return
        with open(save_path, "wb") as f:
            f.write(data)
        self._set_output_text(
            f"File saved successfully.\n\n"
            f"Path : {save_path}\n"
            f"Size : {_fmt_bytes(len(data))}", SUCCESS)
        self.extract_file_lbl.config(text=f"💾  Saved → {save_path}", fg=SUCCESS)

    def _extract_error(self, msg):
        self.extract_btn.config(state="normal", text="🔓  Extract Message")
        self.extract_status.config(text=f"❌  {msg}", fg=DANGER)
        self._set_output_text(f"❌  EXTRACTION FAILED\n\n{msg}", DANGER)



if __name__ == "__main__":
    root = tk.Tk()
    app  = StegoApp(root)
    root.mainloop()