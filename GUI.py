import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

from src.stego_video import embed_message, extract_message
from src.video_io import read_video_frames, mse_psnr_video, color_histogram_video

# ==============================
# MAIN APP
# ==============================

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography Dashboard")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.video_frames = []
        self.current_frame = 0
        self.playing = False

        self.setup_ui()

    # ==============================
    # UI
    # ==============================

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.embed_tab = tk.Frame(notebook, bg="#1e1e1e")
        self.extract_tab = tk.Frame(notebook, bg="#1e1e1e")

        notebook.add(self.embed_tab, text="Embed")
        notebook.add(self.extract_tab, text="Extract")

        self.build_embed_ui()
        self.build_extract_ui()

    # ==============================
    # EMBED UI
    # ==============================

    def build_embed_ui(self):
        frame = self.embed_tab

        tk.Button(frame, text="Select Video", command=self.select_video).pack()
        self.video_label = tk.Label(frame, text="No video selected", bg="#1e1e1e", fg="white")
        self.video_label.pack()

        tk.Label(frame, text="Secret Message", bg="#1e1e1e", fg="white").pack()
        self.message_entry = tk.Text(frame, height=5)
        self.message_entry.pack()

        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="Use Encryption", variable=self.encrypt_var).pack()

        self.key_entry = tk.Entry(frame)
        self.key_entry.pack()

        self.random_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="Random Mode", variable=self.random_var).pack()

        self.stego_key_entry = tk.Entry(frame)
        self.stego_key_entry.pack()

        tk.Button(frame, text="Embed", command=self.run_embed).pack(pady=10)

        self.metrics_label = tk.Label(frame, text="", bg="#1e1e1e", fg="white")
        self.metrics_label.pack()

        # video canvas
        self.canvas = tk.Label(frame)
        self.canvas.pack()

        controls = tk.Frame(frame, bg="#1e1e1e")
        controls.pack()

        tk.Button(controls, text="Play", command=self.play_video).pack(side="left")
        tk.Button(controls, text="Pause", command=self.pause_video).pack(side="left")

    # ==============================
    # EXTRACT UI
    # ==============================

    def build_extract_ui(self):
        frame = self.extract_tab

        tk.Button(frame, text="Select Stego Video", command=self.select_stego).pack()

        self.extract_label = tk.Label(frame, text="", bg="#1e1e1e", fg="white")
        self.extract_label.pack()

        self.extract_key = tk.Entry(frame)
        self.extract_key.pack()

        self.extract_stego_key = tk.Entry(frame)
        self.extract_stego_key.pack()

        tk.Button(frame, text="Extract", command=self.run_extract).pack(pady=10)

        self.output_text = tk.Text(frame, height=10)
        self.output_text.pack()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
        self.video_label.config(text=self.video_path)
        #load videonya (kyk media player)
        if self.video_path:
            self.load_video(self.video_path)
            self.play_video()

    def run_embed(self):
        try:
            msg = self.message_entry.get("1.0", tk.END).encode()

            key = int(self.key_entry.get(), 0) if self.encrypt_var.get() else None
            stego_key = int(self.stego_key_entry.get()) if self.random_var.get() else None

            output_path = filedialog.asksaveasfilename(defaultextension=".avi")

            result = embed_message(
                cover_path=self.video_path,
                output_path=output_path,
                message=msg,
                is_text=True,
                use_encryption=self.encrypt_var.get(),
                a51_key=key,
                use_random=self.random_var.get(),
                stego_key=stego_key
            )

            self.metrics_label.config(
                text=f"MSE: {result['mse_avg']:.4f} | PSNR: {result['psnr_avg']:.2f}"
            )

            self.load_video(output_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def select_stego(self):
        self.stego_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])

    def run_extract(self):
        try:
            key = int(self.extract_key.get(), 0) if self.extract_key.get() else None
            stego_key = int(self.extract_stego_key.get()) if self.extract_stego_key.get() else None

            result = extract_message(
                stego_path=self.stego_path,
                a51_key=key,
                stego_key=stego_key
            )

            if result["is_text"]:
                self.output_text.insert(tk.END, result["message"].decode())
            else:
                save_path = filedialog.asksaveasfilename(initialfile=result["filename"])
                with open(save_path, "wb") as f:
                    f.write(result["message"])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_video(self, path):
        self.video_frames, self.fps = read_video_frames(path)
        self.current_frame = 0
        self.playing = False

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_frame(self):
        if not self.playing:
            return

        frame = self.video_frames[self.current_frame]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)

        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.current_frame = (self.current_frame + 1) % len(self.video_frames)

        self.root.after(int(1000/self.fps), self.update_frame)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()