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

# Import based on format - will select at runtime in embed function
from src.stego_video import embed_message as embed_avi, extract_message as extract_avi
try:
    from src.stego_video_mp4 import embed_message as embed_mp4, extract_message as extract_mp4
except ImportError:
    embed_mp4 = embed_avi
    extract_mp4 = extract_avi

from src.video_io_mp4 import read_video_frames, get_format, mse_psnr_video
from src.video_io import color_histogram_video  # For histogram only

# ─── PALETTE ──────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL     = "#16213e"
CARD      = "#0f3460"
ACCENT    = "#e94560"
ACCENT2   = "#533483"
TEXT      = "#eaeaea"
MUTED     = "#8892a4"
SUCCESS   = "#4ecca3"
WARNING   = "#f5a623"
BTN_BG    = "#0f3460"
BTN_FG    = "#eaeaea"
BTN_ACT   = "#e94560"

DISPLAY_W = 420
DISPLAY_H = 240


def _resize_for_display(frame: np.ndarray, w=DISPLAY_W, h=DISPLAY_H) -> ImageTk.PhotoImage:
    """Resize a BGR numpy frame to display size and return an ImageTk.PhotoImage."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


def _make_btn(parent, text, command, bg=BTN_BG, fg=BTN_FG, **kw):
    return tk.Button(parent, text=text, command=command,
                     bg=bg, fg=fg, activebackground=BTN_ACT, activeforeground="white",
                     relief="flat", padx=10, pady=4, cursor="hand2", **kw)


def _make_label(parent, text="", fg=TEXT, font=None, **kw):
    return tk.Label(parent, text=text, bg=kw.pop("bg", PANEL),
                    fg=fg, font=font or ("Segoe UI", 9), **kw)


# ═══════════════════════════════════════════════════════════════════════════════
class StegoApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography Dashboard")
        self.root.geometry("1280x760")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # ── embed-tab state
        self.video_frames   = []
        self.fps            = 30
        self.current_frame  = 0
        self.playing        = False
        self.video_selected = False
        self.video_path     = None
        self.video_format   = None  # "avi" or "mp4"
        self._play_job      = None

        # ── compare-tab state
        self.cmp_cover_frames = []
        self.cmp_stego_frames = []
        self.cmp_cover_path   = None
        self.cmp_stego_path   = None
        self.cmp_idx          = 0
        self.cmp_mse_list     = []
        self.cmp_psnr_list    = []

        self._apply_style()
        self._setup_ui()

    # ── STYLE ─────────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",        background=BG,    borderwidth=0)
        s.configure("TNotebook.Tab",    background=CARD,  foreground=MUTED,
                    padding=[16, 6],    font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT2)],
              foreground=[("selected", TEXT)])
        s.configure("TScale",           background=BG,    troughcolor=PANEL)
        s.configure("Horizontal.TScale", background=BG)

    # ── NOTEBOOK ──────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.embed_tab   = tk.Frame(self.notebook, bg=BG)
        self.extract_tab = tk.Frame(self.notebook, bg=BG)
        self.compare_tab = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.embed_tab,   text="  📥  Embed  ")
        self.notebook.add(self.extract_tab, text="  📤  Extract  ")
        self.notebook.add(self.compare_tab, text="  🔍  Compare  ")

        self._build_embed_ui()
        self._build_extract_ui()
        self._build_compare_ui()

    # ══════════════════════════════════════════════════════════════════════════
    #  EMBED TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_embed_ui(self):
        root = self.embed_tab

        root.columnconfigure(0, weight=1, minsize=320)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(2, weight=1)

        # ── LEFT: controls ───────────────────────────────────────────────────
        left = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        left.columnconfigure(0, weight=1)

        _make_label(left, "VIDEO SOURCE", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        self.embed_select_btn = _make_btn(left, "📂  Select Cover Video",
                                          self._toggle_video, bg=CARD)
        self.embed_select_btn.grid(row=1, column=0, sticky="ew")

        self.embed_video_lbl = _make_label(left, "No video selected",
                                           fg=MUTED, bg=PANEL,
                                           font=("Segoe UI", 8))
        self.embed_video_lbl.grid(row=2, column=0, sticky="w", pady=(2, 0))

        self.embed_format_lbl = _make_label(left, "",
                                            fg=SUCCESS, bg=PANEL,
                                            font=("Segoe UI", 7))
        self.embed_format_lbl.grid(row=2, column=0, sticky="e", pady=(2, 0))

        _make_label(left, "MESSAGE", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=3, column=0, sticky="w", pady=(10, 0))
        _make_label(left, "Leave blank to embed a file instead",
                    fg=MUTED, font=("Segoe UI", 7), bg=PANEL).grid(
            row=4, column=0, sticky="w", pady=(0, 4))

        self.message_entry = tk.Text(left, height=5, bg=CARD, fg=TEXT,
                                     insertbackground=TEXT,
                                     font=("Consolas", 9), relief="flat",
                                     padx=6, pady=4)
        self.message_entry.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        # ── Encryption ───────────────────────────────────────────────────────
        _make_label(left, "ENCRYPTION (A5/1)", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=6, column=0, sticky="w")

        enc_row = tk.Frame(left, bg=PANEL)
        enc_row.grid(row=7, column=0, sticky="ew", pady=(2, 2))

        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(enc_row, text="Enable A5/1 Encryption",
                       variable=self.encrypt_var,
                       bg=PANEL, fg=TEXT, selectcolor=CARD,
                       activebackground=PANEL, font=("Segoe UI", 9),
                       command=self._toggle_encrypt_ui).pack(side="left")

        self.key_entry = tk.Entry(left, bg=CARD, fg=TEXT,
                                  insertbackground=TEXT, relief="flat",
                                  font=("Consolas", 9))
        self.key_entry.insert(0, "e.g. 0x123456789ABCDEF0")
        self.key_entry.config(fg=MUTED)
        self.key_entry.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        self._bind_placeholder(self.key_entry, "e.g. 0x123456789ABCDEF0")

        # ── Random embed ─────────────────────────────────────────────────────
        _make_label(left, "RANDOM EMBEDDING", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=9, column=0, sticky="w")

        rand_row = tk.Frame(left, bg=PANEL)
        rand_row.grid(row=10, column=0, sticky="ew", pady=(2, 2))

        self.random_var = tk.BooleanVar()
        tk.Checkbutton(rand_row, text="Enable Random Pixel Order",
                       variable=self.random_var,
                       bg=PANEL, fg=TEXT, selectcolor=CARD,
                       activebackground=PANEL, font=("Segoe UI", 9),
                       command=self._toggle_random_ui).pack(side="left")

        self.stego_key_entry = tk.Entry(left, bg=CARD, fg=TEXT,
                                        insertbackground=TEXT, relief="flat",
                                        font=("Consolas", 9))
        self.stego_key_entry.insert(0, "Stego key (integer)")
        self.stego_key_entry.config(fg=MUTED)
        self.stego_key_entry.grid(row=11, column=0, sticky="ew", pady=(0, 14))
        self._bind_placeholder(self.stego_key_entry, "Stego key (integer)")

        self.embed_btn = _make_btn(left, "🔒  Embed Message",
                                   self._run_embed,
                                   bg=ACCENT, fg="white",
                                   font=("Segoe UI", 10, "bold"))
        self.embed_btn.grid(row=12, column=0, sticky="ew", ipady=4)

        self.embed_status = _make_label(left, "", fg=MUTED,
                                        font=("Segoe UI", 8), bg=PANEL)
        self.embed_status.grid(row=13, column=0, sticky="w", pady=(4, 0))

        # ── RIGHT: video preview ──────────────────────────────────────────────
        right = tk.Frame(root, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        _make_label(right, "VIDEO PREVIEW", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=0, column=0, pady=(8, 4))

        self.embed_canvas = tk.Label(right, bg="#000",
                                     width=DISPLAY_W, height=DISPLAY_H)
        self.embed_canvas.grid(row=1, column=0, padx=8)

        ctrl = tk.Frame(right, bg=PANEL)
        ctrl.grid(row=2, column=0, pady=6)
        _make_btn(ctrl, "▶  Play",  self._play_video,  bg=CARD).pack(side="left", padx=4)
        _make_btn(ctrl, "⏸  Pause", self._pause_video, bg=CARD).pack(side="left", padx=4)

        # ── BOTTOM: metrics ───────────────────────────────────────────────────
        metrics = tk.Frame(root, bg=CARD, pady=6)
        metrics.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6)

        self.metrics_label = _make_label(metrics,
                                         "MSE: —  |  PSNR: —  |  Capacity: —",
                                         fg=SUCCESS, font=("Consolas", 10),
                                         bg=CARD)
        self.metrics_label.pack()

        self.key_saved_label = _make_label(metrics, "", fg=WARNING,
                                           font=("Segoe UI", 8), bg=CARD)
        self.key_saved_label.pack()

        # ── BOTTOM: histogram chart ───────────────────────────────────────────
        self.chart_frame = tk.Frame(root, bg=BG)
        self.chart_frame.grid(row=2, column=0, columnspan=2,
                               sticky="nsew", padx=6, pady=4)

    # ── placeholder helpers ───────────────────────────────────────────────────
    def _bind_placeholder(self, entry, placeholder):
        def on_focus_in(e):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(fg=TEXT)
        def on_focus_out(e):
            if not entry.get():
                entry.insert(0, placeholder)
                entry.config(fg=MUTED)
        entry.bind("<FocusIn>",  on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    def _toggle_encrypt_ui(self):
        pass  # visual state managed by BooleanVar

    def _toggle_random_ui(self):
        pass

    def _get_key(self, entry, placeholder):
        """Return stripped entry value or None if it's empty / placeholder."""
        v = entry.get().strip()
        return None if (not v or v == placeholder) else v

    # ── video toggle ─────────────────────────────────────────────────────────
    def _toggle_video(self):
        if not self.video_selected:
            path = filedialog.askopenfilename(
                title="Select Cover Video",
                filetypes=[("Video files", "*.avi *.mp4"),
                          ("AVI video", "*.avi"),
                          ("MP4 video", "*.mp4"),
                          ("All files", "*.*")])
            if not path:
                return
            self.video_path = path
            self.embed_video_lbl.config(
                text=os.path.basename(path), fg=TEXT)
            self._load_video(path)
            self._play_video()
            self.video_selected = True
            self.embed_select_btn.config(text="✖  Clear Video")
        else:
            if self._play_job:
                self.root.after_cancel(self._play_job)
                self._play_job = None
            self.playing        = False
            self.video_path     = None
            self.video_format   = None
            self.video_frames   = []
            self.video_selected = False
            self.embed_canvas.config(image="")
            self.embed_video_lbl.config(text="No video selected", fg=MUTED)
            self.embed_format_lbl.config(text="")
            self.embed_select_btn.config(text="📂  Select Cover Video")

    def _load_video(self, path):
        self.video_frames, self.fps = read_video_frames(path)
        self.video_format = get_format(path).upper()  # "AVI" or "MP4"
        self.embed_format_lbl.config(text=f"[{self.video_format}]", fg=SUCCESS)

    def _play_video(self):
        self.playing = True
        self._update_frame()

    def _pause_video(self):
        self.playing = False

    def _update_frame(self):
        if not self.playing or not self.video_frames:
            return
        frame = self.video_frames[self.current_frame]
        img   = _resize_for_display(frame)
        self.embed_canvas.configure(image=img)
        self.embed_canvas.image = img
        self.current_frame = (self.current_frame + 1) % len(self.video_frames)
        self._play_job = self.root.after(int(1000 / max(self.fps, 1)),
                                         self._update_frame)

    # ── embed (main thread: validate + collect dialogs; worker thread: heavy work)
    def _run_embed(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a cover video first.")
            return

        # Collect message / file on main thread
        text = self.message_entry.get("1.0", tk.END).strip()
        if text:
            msg       = text.encode("utf-8")
            is_text   = True
            extension = ".txt"
            filename  = "message.txt"
        else:
            file_path = filedialog.askopenfilename(title="Select file to embed")
            if not file_path:
                return
            with open(file_path, "rb") as f:
                msg = f.read()
            is_text   = False
            extension = os.path.splitext(file_path)[1]
            filename  = os.path.basename(file_path)

        # Validate & parse keys on main thread
        use_encrypt = self.encrypt_var.get()
        use_random  = self.random_var.get()
        a51_key     = None
        stego_key   = None

        if use_encrypt:
            raw = self._get_key(self.key_entry, "e.g. 0x123456789ABCDEF0")
            if not raw:
                # Generate random A5/1 key (64-bit)
                import random
                a51_key = random.getrandbits(64)
                messagebox.showinfo("Info", f"Generated random A5/1 key:\n0x{a51_key:016x}")
            else:
                try:
                    a51_key = int(raw, 0)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid A5/1 key: '{raw}'")
                    return

        if use_random:
            raw = self._get_key(self.stego_key_entry, "Stego key (integer)")
            if not raw:
                # Generate random stego key (32-bit seed)
                import random
                stego_key = random.getrandbits(32)
                messagebox.showinfo("Info", f"Generated random stego key:\n{stego_key}")
            else:
                try:
                    stego_key = int(raw)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid stego key: '{raw}'")
                    return

        # Output path on main thread
        # Default to same format as input
        if self.video_format == "MP4":
            default_ext = ".mp4"
            filetypes = [("MP4 video", "*.mp4"), ("AVI video", "*.avi")]
        else:
            default_ext = ".avi"
            filetypes = [("AVI video", "*.avi"), ("MP4 video", "*.mp4")]
        
        output_path = filedialog.asksaveasfilename(
            title="Save stego video as",
            defaultextension=default_ext,
            filetypes=filetypes)
        if not output_path:
            return

        # Disable button and hand off to worker thread
        self.embed_btn.config(state="disabled", text="⏳  Embedding…")
        self.embed_status.config(text="Processing… please wait.", fg=WARNING)
        self.key_saved_label.config(text="")
        self.metrics_label.config(text="MSE: —  |  PSNR: —  |  Capacity: —")

        thread = threading.Thread(
            target=self._embed_worker,
            args=(msg, is_text, extension, filename,
                  use_encrypt, a51_key, use_random, stego_key,
                  output_path),
            daemon=True
        )
        thread.start()

    def _embed_worker(self, msg, is_text, extension, filename,
                      use_encrypt, a51_key, use_random, stego_key,
                      output_path):
        try:
            # Select appropriate embed function based on output format
            output_format = get_format(output_path).lower()
            embed_func = embed_mp4 if output_format == "mp4" else embed_avi
            
            result = embed_func(
                cover_path     = self.video_path,
                output_path    = output_path,
                message        = msg,
                is_text        = is_text,
                extension      = extension,
                filename       = filename,
                use_encryption = use_encrypt,
                a51_key        = a51_key,
                use_random     = use_random,
                stego_key      = stego_key
            )

            # Re-read stego frames for histogram (in worker thread — non-blocking)
            cover_frames  = self.video_frames
            stego_frames, _ = read_video_frames(output_path)

            key_path = None
            if use_encrypt or use_random:
                key_path = self._save_key_file(output_path, a51_key, stego_key)

            self.root.after(0, self._embed_done,
                            result, cover_frames, stego_frames, key_path)
        except Exception as exc:
            self.root.after(0, self._embed_error, str(exc))

    def _embed_done(self, result, cover_frames, stego_frames, key_path):
        self.embed_btn.config(state="normal", text="🔒  Embed Message")
        
        cap_kb = result["total_capacity_bytes"] // 1024
        fmt_str = result.get("format", "AVI")  # Get format from result
        
        self.metrics_label.config(
            text=(f"Format: {fmt_str}  |  MSE: {result['mse_avg']:.4f}  |  "
                  f"PSNR: {result['psnr_avg']:.2f} dB  |  "
                  f"Capacity: {cap_kb} KB"),
            fg=SUCCESS
        )

        if key_path:
            # Show file path prominently
            key_display = f"🔑  Keys saved to: {os.path.basename(key_path)}"
            self.key_saved_label.config(text=key_display, fg=SUCCESS)
            self.embed_status.config(
                text=f"✅  Embedding complete!  (Keys: {os.path.basename(key_path)})", 
                fg=SUCCESS
            )
            # Also show full path in a messagebox
            full_path = os.path.abspath(key_path)
            messagebox.showinfo("Keys Saved", 
                f"Encryption & steganography keys saved to:\n\n{full_path}\n\n"
                "Keep this file safe! You need it to extract the message.")
        else:
            self.embed_status.config(text="✅  Embedding complete!", fg=SUCCESS)

        self._show_histogram(cover_frames, stego_frames, result["mse_list"])

    def _embed_error(self, msg):
        self.embed_btn.config(state="normal", text="🔒  Embed Message")
        self.embed_status.config(text=f"❌  {msg}", fg=ACCENT)
        messagebox.showerror("Embed Error", msg)

    def _save_key_file(self, output_path, a51_key, stego_key):
        """Save encryption/stego keys to a text file next to the output video."""
        base     = os.path.splitext(output_path)[0]
        key_path = base + "_keys.txt"
        ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=== Video Steganography — Key File ===",
            f"Generated : {ts}",
            f"Video     : {os.path.basename(output_path)}",
            "",
        ]
        if a51_key is not None:
            lines.append(f"A5/1 Key  : {hex(a51_key)}")
        if stego_key is not None:
            lines.append(f"Stego Key : {stego_key}")
        lines += [
            "",
            "KEEP THIS FILE SAFE.",
            "You will need these keys to extract the hidden message.",
        ]
        with open(key_path, "w") as fh:
            fh.write("\n".join(lines))
        return key_path

    def _show_histogram(self, cover_frames, stego_frames, mse_list):
        hb_c, hg_c, hr_c = color_histogram_video(cover_frames)
        hb_s, hg_s, hr_s = color_histogram_video(stego_frames)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), facecolor="#0f3460")
        fig.suptitle(f"Color Histogram — Cover vs Stego   (MSE avg = {np.mean(mse_list):.4f})",
                     fontsize=9, color=TEXT)

        for ax, (hc, hs, ch) in zip(
                axes,
                [(hb_c, hb_s, "#4e9af1"),
                 (hg_c, hg_s, "#4ecca3"),
                 (hr_c, hr_s, "#e94560")]):
            ax.plot(hc, color=ch,        alpha=0.8, linewidth=1,   label="Cover")
            ax.plot(hs, color=ch,        alpha=0.5, linewidth=1,
                    linestyle="--",  label="Stego")
            ax.set_facecolor("#16213e")
            ax.tick_params(colors=MUTED, labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(CARD)
        axes[0].set_title("Blue",  color=TEXT, fontsize=8)
        axes[1].set_title("Green", color=TEXT, fontsize=8)
        axes[2].set_title("Red",   color=TEXT, fontsize=8)
        axes[1].legend(fontsize=6, facecolor=CARD, labelcolor=TEXT)
        fig.tight_layout()

        for w in self.chart_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXTRACT TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_extract_ui(self):
        root = self.extract_tab
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # ── LEFT: controls ────────────────────────────────────────────────────
        left = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        left.columnconfigure(0, weight=1)

        _make_label(left, "STEGO VIDEO", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        _make_btn(left, "📂  Select Stego Video",
                  self._select_stego, bg=CARD).grid(row=1, column=0, sticky="ew")

        self.stego_lbl = _make_label(left, "No file selected",
                                     fg=MUTED, font=("Segoe UI", 8), bg=PANEL)
        self.stego_lbl.grid(row=2, column=0, sticky="w", pady=(2, 0))

        self.stego_format_lbl = _make_label(left, "",
                                            fg=SUCCESS, bg=PANEL,
                                            font=("Segoe UI", 7))
        self.stego_format_lbl.grid(row=2, column=0, sticky="e", pady=(2, 0))

        _make_label(left, "A5/1 KEY", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=3, column=0, sticky="w")
        _make_label(left, "Leave blank if encryption was not used",
                    fg=MUTED, font=("Segoe UI", 7), bg=PANEL).grid(
            row=4, column=0, sticky="w", pady=(0, 2))
        self.extract_key = tk.Entry(left, bg=CARD, fg=TEXT,
                                    insertbackground=TEXT, relief="flat",
                                    font=("Consolas", 9))
        self.extract_key.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        _make_label(left, "STEGO KEY", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=6, column=0, sticky="w")
        _make_label(left, "Leave blank if random mode was not used",
                    fg=MUTED, font=("Segoe UI", 7), bg=PANEL).grid(
            row=7, column=0, sticky="w", pady=(0, 2))
        self.extract_stego_key = tk.Entry(left, bg=CARD, fg=TEXT,
                                          insertbackground=TEXT, relief="flat",
                                          font=("Consolas", 9))
        self.extract_stego_key.grid(row=8, column=0, sticky="ew", pady=(0, 14))

        self.extract_btn = _make_btn(left, "🔓  Extract Message",
                                     self._run_extract,
                                     bg=ACCENT, fg="white",
                                     font=("Segoe UI", 10, "bold"))
        self.extract_btn.grid(row=9, column=0, sticky="ew", ipady=4)

        self.extract_status = _make_label(left, "", fg=MUTED,
                                          font=("Segoe UI", 8), bg=PANEL)
        self.extract_status.grid(row=10, column=0, sticky="w", pady=(4, 0))

        # file-saved indicator
        self.extract_file_lbl = _make_label(left, "", fg=WARNING,
                                            font=("Segoe UI", 8), bg=PANEL)
        self.extract_file_lbl.grid(row=11, column=0, sticky="w", pady=(2, 0))

        # ── RIGHT: output display ─────────────────────────────────────────────
        right = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        _make_label(right, "EXTRACTED MESSAGE", fg=ACCENT,
                    font=("Segoe UI", 8, "bold"), bg=PANEL).grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        self.output_text = tk.Text(
            right, bg=CARD, fg=SUCCESS,
            insertbackground=TEXT, relief="flat",
            font=("Consolas", 10), padx=8, pady=8,
            wrap="word", state="disabled"
        )
        self.output_text.grid(row=1, column=0, sticky="nsew")

        sb = tk.Scrollbar(right, command=self.output_text.yview, bg=PANEL)
        sb.grid(row=1, column=1, sticky="ns")
        self.output_text.config(yscrollcommand=sb.set)

    def _select_stego(self):
        path = filedialog.askopenfilename(
            title="Select Stego Video",
            filetypes=[("Video files", "*.avi *.mp4"),
                      ("AVI video", "*.avi"),
                      ("MP4 video", "*.mp4"),
                      ("All files", "*.*")])
        if path:
            self.stego_path = path
            stego_format = get_format(path).upper()
            self.stego_lbl.config(text=os.path.basename(path), fg=TEXT)
            self.stego_format_lbl.config(text=f"[{stego_format}]", fg=SUCCESS)

    def _set_output_text(self, text, color=SUCCESS):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled", fg=color)

    def _run_extract(self):
        if not hasattr(self, "stego_path") or not self.stego_path:
            messagebox.showerror("Error", "Please select a stego video first.")
            return

        raw_key   = self.extract_key.get().strip()
        raw_sk    = self.extract_stego_key.get().strip()
        a51_key   = None
        stego_key = None

        if raw_key:
            try:
                a51_key = int(raw_key, 0)
            except ValueError:
                messagebox.showerror("Error", f"Invalid A5/1 key: '{raw_key}'")
                return
        if raw_sk:
            try:
                stego_key = int(raw_sk)
            except ValueError:
                messagebox.showerror("Error", f"Invalid stego key: '{raw_sk}'")
                return

        self.extract_btn.config(state="disabled", text="⏳  Extracting…")
        self.extract_status.config(text="Processing… please wait.", fg=WARNING)
        self.extract_file_lbl.config(text="")
        self._set_output_text("⏳  Extracting message…", WARNING)

        thread = threading.Thread(
            target=self._extract_worker,
            args=(self.stego_path, a51_key, stego_key),
            daemon=True
        )
        thread.start()

    def _extract_worker(self, stego_path, a51_key, stego_key):
        try:
            # Select appropriate extract function based on stego video format
            stego_format = get_format(stego_path).lower()
            extract_func = extract_mp4 if stego_format == "mp4" else extract_avi
            
            result = extract_func(stego_path, a51_key, stego_key)
            self.root.after(0, self._extract_done, result, stego_path)
        except Exception as exc:
            self.root.after(0, self._extract_error, str(exc))

    def _extract_done(self, result, stego_path):
        self.extract_btn.config(state="normal", text="🔓  Extract Message")
        self.extract_status.config(text="✅  Extraction complete!", fg=SUCCESS)

        os.makedirs("tests_output", exist_ok=True)
        base = os.path.splitext(os.path.basename(stego_path))[0]

        if result["is_text"]:
            try:
                text = result["message"].decode("utf-8")
            except UnicodeDecodeError:
                text = result["message"].decode("latin-1", errors="replace")

            display = (
                f"{'─'*60}\n"
                f"  Format    : {result.get('format', 'UNKNOWN')}\n"
                f"  Mode      : {'Random' if result['is_random'] else 'Sequential'}\n"
                f"  Encrypted : {'Yes' if result['is_encrypted'] else 'No'}\n"
                f"  Size      : {result['payload_size']} bytes\n"
                f"{'─'*60}\n\n"
                f"{text}"
            )
            self._set_output_text(display, SUCCESS)

            out_path = f"tests_output/{base}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            self.extract_file_lbl.config(
                text=f"💾  Saved → {out_path}", fg=WARNING)
        else:
            ext      = result["extension"] or ".bin"
            out_path = f"tests_output/{base}{ext}"
            with open(out_path, "wb") as f:
                f.write(result["message"])

            display = (
                f"{'─'*60}\n"
                f"  Mode      : {'Random' if result['is_random'] else 'Sequential'}\n"
                f"  Encrypted : {'Yes' if result['is_encrypted'] else 'No'}\n"
                f"  File type : {ext}\n"
                f"  Size      : {result['payload_size']} bytes\n"
                f"{'─'*60}\n\n"
                f"Binary file extracted.\n"
                f"Saved to: {out_path}"
            )
            self._set_output_text(display, WARNING)
            self.extract_file_lbl.config(
                text=f"💾  Saved → {out_path}", fg=WARNING)

    def _extract_error(self, msg):
        self.extract_btn.config(state="normal", text="🔓  Extract Message")
        self.extract_status.config(text=f"❌  {msg}", fg=ACCENT)
        self._set_output_text(f"❌  EXTRACTION FAILED\n\n{msg}", ACCENT)

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPARE TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_compare_ui(self):
        root = self.compare_tab
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(3, weight=1)

        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(root, bg=PANEL, padx=10, pady=8)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))

        _make_btn(top, "📂  Load Cover Video",
                  self._load_compare_cover, bg=CARD).pack(side="left", padx=4)
        self.cmp_cover_lbl = _make_label(top, "No cover loaded",
                                         fg=MUTED, bg=PANEL,
                                         font=("Segoe UI", 8))
        self.cmp_cover_lbl.pack(side="left", padx=6)

        _make_btn(top, "📂  Load Stego Video",
                  self._load_compare_stego, bg=CARD).pack(side="left", padx=(20, 4))
        self.cmp_stego_lbl = _make_label(top, "No stego loaded",
                                         fg=MUTED, bg=PANEL,
                                         font=("Segoe UI", 8))
        self.cmp_stego_lbl.pack(side="left", padx=6)

        self.cmp_calc_btn = _make_btn(top, "📊  Calculate MSE & PSNR",
                                      self._run_compare,
                                      bg=ACCENT, fg="white",
                                      font=("Segoe UI", 9, "bold"))
        self.cmp_calc_btn.pack(side="right", padx=4)

        # ── video panels ─────────────────────────────────────────────────────
        def _video_panel(parent, col, title_var_name, canvas_var_name, title_text):
            panel = tk.Frame(parent, bg=PANEL)
            panel.grid(row=1, column=col, sticky="nsew", padx=(6 if col == 0 else 3, 3 if col == 0 else 6), pady=2)
            panel.columnconfigure(0, weight=1)
            lbl = _make_label(panel, title_text, fg=ACCENT,
                              font=("Segoe UI", 8, "bold"), bg=PANEL)
            lbl.grid(row=0, column=0, pady=(6, 2))
            setattr(self, title_var_name, lbl)
            canvas = tk.Label(panel, bg="#000",
                              width=DISPLAY_W, height=DISPLAY_H)
            canvas.grid(row=1, column=0, padx=8, pady=4)
            setattr(self, canvas_var_name, canvas)
            return panel

        _video_panel(root, 0, "_cmp_cover_title", "cmp_cover_canvas", "COVER VIDEO")
        _video_panel(root, 1, "_cmp_stego_title", "cmp_stego_canvas", "STEGO VIDEO")

        # ── frame scrub slider ───────────────────────────────────────────────
        slider_row = tk.Frame(root, bg=BG, pady=4)
        slider_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        slider_row.columnconfigure(1, weight=1)

        _make_label(slider_row, "Frame:", fg=TEXT, bg=BG).grid(row=0, column=0, padx=(0, 6))
        self.cmp_slider = ttk.Scale(slider_row, from_=0, to=0,
                                    orient="horizontal",
                                    command=self._compare_scrub)
        self.cmp_slider.grid(row=0, column=1, sticky="ew")
        self.cmp_frame_lbl = _make_label(slider_row, "0 / 0",
                                         fg=TEXT, bg=BG,
                                         font=("Segoe UI", 8))
        self.cmp_frame_lbl.grid(row=0, column=2, padx=(6, 0))

        # ── stats bar ────────────────────────────────────────────────────────
        stats = tk.Frame(root, bg=CARD, pady=6)
        stats.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
        stats.columnconfigure(0, weight=1)

        self.cmp_stats_lbl = _make_label(
            stats,
            "Load both videos, then click  📊 Calculate MSE & PSNR",
            fg=MUTED, font=("Consolas", 10), bg=CARD)
        self.cmp_stats_lbl.grid(row=0, column=0, pady=4)

        # ── chart area ───────────────────────────────────────────────────────
        self.cmp_chart_frame = tk.Frame(root, bg=BG)
        self.cmp_chart_frame.grid(row=4, column=0, columnspan=2,
                                   sticky="nsew", padx=6, pady=(2, 6))
        root.rowconfigure(4, weight=1)

    # ── compare helpers ───────────────────────────────────────────────────────
    def _load_compare_cover(self):
        path = filedialog.askopenfilename(
            title="Select Cover Video",
            filetypes=[("Video files", "*.avi *.mp4")])
        if not path:
            return
        self.cmp_cover_path = path
        self.cmp_cover_lbl.config(text=os.path.basename(path), fg=TEXT)
        self.cmp_calc_btn.config(state="disabled", text="⏳  Loading…")
        threading.Thread(target=self._load_cmp_cover_worker,
                         args=(path,), daemon=True).start()

    def _load_cmp_cover_worker(self, path):
        frames, _ = read_video_frames(path)
        self.cmp_cover_frames = frames
        self.root.after(0, self._cmp_after_load)

    def _load_compare_stego(self):
        path = filedialog.askopenfilename(
            title="Select Stego Video",
            filetypes=[("Video files", "*.avi *.mp4")])
        if not path:
            return
        self.cmp_stego_path = path
        self.cmp_stego_lbl.config(text=os.path.basename(path), fg=TEXT)
        self.cmp_calc_btn.config(state="disabled", text="⏳  Loading…")
        threading.Thread(target=self._load_cmp_stego_worker,
                         args=(path,), daemon=True).start()

    def _load_cmp_stego_worker(self, path):
        frames, _ = read_video_frames(path)
        self.cmp_stego_frames = frames
        self.root.after(0, self._cmp_after_load)

    def _cmp_after_load(self):
        self.cmp_calc_btn.config(state="normal", text="📊  Calculate MSE & PSNR")
        n = min(len(self.cmp_cover_frames), len(self.cmp_stego_frames))
        if n > 0:
            self.cmp_slider.config(to=max(0, n - 1))
            self.cmp_idx = 0
            self._show_cmp_frame(0)

    def _compare_scrub(self, val):
        idx = int(float(val))
        self.cmp_idx = idx
        self._show_cmp_frame(idx)

    def _show_cmp_frame(self, idx):
        n_cover = len(self.cmp_cover_frames)
        n_stego = len(self.cmp_stego_frames)
        total   = min(n_cover, n_stego)

        if n_cover > 0 and idx < n_cover:
            img = _resize_for_display(self.cmp_cover_frames[idx])
            self.cmp_cover_canvas.configure(image=img)
            self.cmp_cover_canvas.image = img

        if n_stego > 0 and idx < n_stego:
            img = _resize_for_display(self.cmp_stego_frames[idx])
            self.cmp_stego_canvas.configure(image=img)
            self.cmp_stego_canvas.image = img

        # Per-frame PSNR overlay
        psnr_str = ""
        if self.cmp_psnr_list and idx < len(self.cmp_psnr_list):
            p = self.cmp_psnr_list[idx]
            m = self.cmp_mse_list[idx]
            psnr_str = f"   Frame {idx}: MSE={m:.4f}  PSNR={p:.2f} dB"

        self.cmp_frame_lbl.config(
            text=f"{idx} / {total - 1 if total > 0 else 0}{psnr_str}")

    def _run_compare(self):
        if not self.cmp_cover_frames or not self.cmp_stego_frames:
            messagebox.showerror("Error",
                                 "Load both a cover and a stego video first.")
            return

        n_c = len(self.cmp_cover_frames)
        n_s = len(self.cmp_stego_frames)
        if n_c != n_s:
            if not messagebox.askyesno(
                    "Frame count mismatch",
                    f"Cover has {n_c} frames, stego has {n_s}.\n"
                    "Compare using the shorter length?"):
                return

        self.cmp_calc_btn.config(state="disabled", text="⏳  Calculating…")
        self.cmp_stats_lbl.config(text="Calculating… please wait.", fg=WARNING)

        threading.Thread(target=self._compare_worker, daemon=True).start()

    def _compare_worker(self):
        try:
            n = min(len(self.cmp_cover_frames), len(self.cmp_stego_frames))
            cover  = self.cmp_cover_frames[:n]
            stego  = self.cmp_stego_frames[:n]
            mse_list, psnr_list, mse_avg, psnr_avg = mse_psnr_video(cover, stego)
            self.root.after(0, self._compare_done,
                            mse_list, psnr_list, mse_avg, psnr_avg)
        except Exception as exc:
            self.root.after(0, self._compare_error, str(exc))

    def _compare_done(self, mse_list, psnr_list, mse_avg, psnr_avg):
        self.cmp_mse_list  = mse_list
        self.cmp_psnr_list = psnr_list

        self.cmp_calc_btn.config(state="normal", text="📊  Calculate MSE & PSNR")

        finite_psnr = [p for p in psnr_list if np.isfinite(p)]
        psnr_min    = min(finite_psnr) if finite_psnr else float('inf')
        psnr_max    = max(finite_psnr) if finite_psnr else float('inf')

        self.cmp_stats_lbl.config(
            text=(f"MSE avg: {mse_avg:.4f}   |   "
                  f"PSNR avg: {psnr_avg:.2f} dB   |   "
                  f"PSNR min: {psnr_min:.2f} dB   |   "
                  f"PSNR max: {psnr_max:.2f} dB   |   "
                  f"Frames compared: {len(mse_list)}"),
            fg=SUCCESS
        )

        # Refresh current frame display with per-frame annotation
        self._show_cmp_frame(self.cmp_idx)
        self._draw_compare_chart(mse_list, psnr_list)

    def _compare_error(self, msg):
        self.cmp_calc_btn.config(state="normal", text="📊  Calculate MSE & PSNR")
        self.cmp_stats_lbl.config(text=f"❌  {msg}", fg=ACCENT)
        messagebox.showerror("Compare Error", msg)

    def _draw_compare_chart(self, mse_list, psnr_list):
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.8),
                                        facecolor=BG)

        frames = list(range(len(mse_list)))

        # MSE per frame
        ax1.fill_between(frames, mse_list, alpha=0.35, color=ACCENT)
        ax1.plot(frames, mse_list, color=ACCENT, linewidth=1)
        ax1.axhline(np.mean(mse_list), color=WARNING, linewidth=1,
                    linestyle="--", label=f"avg={np.mean(mse_list):.4f}")
        ax1.set_title("MSE per Frame", color=TEXT, fontsize=9)
        ax1.set_xlabel("Frame", color=MUTED, fontsize=7)
        ax1.set_facecolor(PANEL)
        ax1.tick_params(colors=MUTED, labelsize=6)
        ax1.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT)
        for sp in ax1.spines.values():
            sp.set_edgecolor(CARD)

        # PSNR per frame (clip inf to finite max + 5 for display)
        finite = [p for p in psnr_list if np.isfinite(p)]
        p_cap  = (max(finite) + 5) if finite else 60
        psnr_clipped = [min(p, p_cap) for p in psnr_list]
        ax2.fill_between(frames, psnr_clipped, alpha=0.35, color=SUCCESS)
        ax2.plot(frames, psnr_clipped, color=SUCCESS, linewidth=1)
        if finite:
            ax2.axhline(np.mean(finite), color=WARNING, linewidth=1,
                        linestyle="--", label=f"avg={np.mean(finite):.2f} dB")
        ax2.set_title("PSNR per Frame (dB)", color=TEXT, fontsize=9)
        ax2.set_xlabel("Frame", color=MUTED, fontsize=7)
        ax2.set_facecolor(PANEL)
        ax2.tick_params(colors=MUTED, labelsize=6)
        ax2.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT)
        for sp in ax2.spines.values():
            sp.set_edgecolor(CARD)

        fig.tight_layout(pad=1.5)

        for w in self.cmp_chart_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.cmp_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = StegoApp(root)
    root.mainloop()