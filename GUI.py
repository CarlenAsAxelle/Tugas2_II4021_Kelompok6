import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.stego_video import embed_message, extract_message
from src.video_io import read_video_frames, color_histogram_video


class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography Dashboard")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.video_frames = []
        self.current_frame = 0
        self.playing = False
        self.video_selected = False

        self.setup_ui()

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.embed_tab = tk.Frame(notebook, bg="#1e1e1e")
        self.extract_tab = tk.Frame(notebook, bg="#1e1e1e")

        notebook.add(self.embed_tab, text="Embed")
        notebook.add(self.extract_tab, text="Extract")

        self.build_embed_ui()
        self.build_extract_ui()

    def build_embed_ui(self):
        main = tk.Frame(self.embed_tab, bg="#1e1e1e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(2, weight=1)

        self.input_frame = tk.Frame(main, bg="#2a2a2a")
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(self.input_frame, text="Controls", bg="#2a2a2a", fg="white").pack()

        self.select_btn = tk.Button(self.input_frame, text="Select Video", command=self.toggle_video)
        self.select_btn.pack()

        self.video_label = tk.Label(self.input_frame, text="No video", bg="#2a2a2a", fg="white")
        self.video_label.pack()

        tk.Label(self.input_frame, text="Message", bg="#2a2a2a", fg="white").pack()
        self.message_entry = tk.Text(self.input_frame, height=5)
        self.message_entry.pack()

        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(self.input_frame, text="Encryption", variable=self.encrypt_var).pack()
        self.key_entry = tk.Entry(self.input_frame)
        self.key_entry.pack()

        self.random_var = tk.BooleanVar()
        tk.Checkbutton(self.input_frame, text="Random Mode", variable=self.random_var).pack()
        self.stego_key_entry = tk.Entry(self.input_frame)
        self.stego_key_entry.pack()

        tk.Button(self.input_frame, text="Embed", command=self.run_embed).pack(pady=10)

        self.video_frame = tk.Frame(main, bg="#000")
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        controls = tk.Frame(self.video_frame, bg="#000")
        controls.pack()

        tk.Button(controls, text="Play", command=self.play_video).pack(side="left")
        tk.Button(controls, text="Pause", command=self.pause_video).pack(side="left")

        self.metrics_frame = tk.Frame(main, bg="#1e1e1e")
        self.metrics_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.metrics_label = tk.Label(self.metrics_frame, text="MSE: - | PSNR: -", fg="white", bg="#1e1e1e")
        self.metrics_label.pack()

        self.chart_frame = tk.Frame(main, bg="#1e1e1e")
        self.chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

    def toggle_video(self):
        if not self.video_selected:
            path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
            if path:
                self.video_path = path
                self.video_label.config(text=path)

                self.load_video(path)
                self.video_frame.grid()
                self.play_video()

                self.video_selected = True
                self.select_btn.config(text="Clear")
        else:
            self.video_path = None
            self.video_label.config(text="No video")
            self.canvas.config(image="")
            self.video_frames = []

            self.video_selected = False
            self.select_btn.config(text="Select Video")


    def show_histogram(self, cover_frames, stego_frames, mse_list):
        hb_c, hg_c, hr_c = color_histogram_video(cover_frames)
        hb_s, hg_s, hr_s = color_histogram_video(stego_frames)

        fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
        fig.suptitle(f"MSE avg={np.mean(mse_list):.4f}", fontsize=10)

        ax[0].plot(hb_c); ax[0].plot(hb_s, linestyle="--"); ax[0].set_title("Blue")
        ax[1].plot(hg_c); ax[1].plot(hg_s, linestyle="--"); ax[1].set_title("Green")
        ax[2].plot(hr_c); ax[2].plot(hr_s, linestyle="--"); ax[2].set_title("Red")

        fig.tight_layout()

        for w in self.chart_frame.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


    def run_embed(self):
        try:
            key = int(self.key_entry.get(), 0) if self.encrypt_var.get() else None
            stego_key = int(self.stego_key_entry.get()) if self.random_var.get() else None

            output_path = filedialog.asksaveasfilename(defaultextension=".avi")

            text = self.message_entry.get("1.0", tk.END).strip()

            if text:
                msg = text.encode()
                is_text = True
                extension = ".txt"
                filename = "message.txt"
            else:
                file_path = filedialog.askopenfilename()
                with open(file_path, "rb") as f:
                    msg = f.read()
                is_text = False
                extension = os.path.splitext(file_path)[1]
                filename = os.path.basename(file_path)

            result = embed_message(
                self.video_path, output_path, msg, is_text,
                extension, filename,
                self.encrypt_var.get(), key,
                self.random_var.get(), stego_key
            )

            self.metrics_label.config(
                text=f"MSE: {result['mse_avg']:.4f} | PSNR: {result['psnr_avg']:.2f}"
            )

            self.video_frame.grid_remove()

            cover_frames, _ = read_video_frames(self.video_path)
            stego_frames, _ = read_video_frames(output_path)

            self.show_histogram(cover_frames, stego_frames, result["mse_list"])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_video(self, path):
        self.video_frames, self.fps = read_video_frames(path)
        self.current_frame = 0

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_frame(self):
        if not self.playing or not self.video_frames:
            return

        frame = self.video_frames[self.current_frame]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=img)
        self.canvas.image = img

        self.current_frame = (self.current_frame + 1) % len(self.video_frames)
        self.root.after(int(1000/self.fps), self.update_frame)

    def build_extract_ui(self):
        frame = self.extract_tab

        self.stego_label = tk.Label(frame, text="No file", bg="#1e1e1e", fg="white")
        self.stego_label.pack()

        tk.Button(frame, text="Select Video", command=self.select_stego).pack()

        tk.Label(frame, text="A5/1 Key", bg="#1e1e1e", fg="white").pack()
        self.extract_key = tk.Entry(frame)
        self.extract_key.pack()

        tk.Label(frame, text="Stego Key", bg="#1e1e1e", fg="white").pack()
        self.extract_stego_key = tk.Entry(frame)
        self.extract_stego_key.pack()

        tk.Button(frame, text="Extract", command=self.run_extract).pack()

        self.output_text = tk.Text(frame, height=10, bg="#111", fg="white")
        self.output_text.pack()

    def set_output_text(self, text):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)

    def reset_extract_ui(self):
        self.stego_label.config(text="No file")
        self.extract_key.delete(0, tk.END)
        self.extract_stego_key.delete(0, tk.END)

    def select_stego(self):
        self.stego_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
        self.stego_label.config(text=self.stego_path)

    def run_extract(self):
        try:
            self.set_output_text("⏳ Extracting...")

            key = int(self.extract_key.get(), 0) if self.extract_key.get() else None
            stego_key = int(self.extract_stego_key.get()) if self.extract_stego_key.get() else None

            result = extract_message(self.stego_path, key, stego_key)

            os.makedirs("tests_output", exist_ok=True)
            base = os.path.splitext(os.path.basename(self.stego_path))[0]

            if result["is_text"]:
                text = result["message"].decode(errors="replace")

                self.set_output_text(f"✅ SUCCESS\n\n{text}")

                with open(f"tests_output/{base}.txt", "w") as f:
                    f.write(text)
            else:
                ext = result["extension"] if result["extension"] else ".bin"
                path = f"tests_output/{base}{ext}"

                with open(path, "wb") as f:
                    f.write(result["message"])

                self.set_output_text(f"✅ File saved:\n{path}")

            self.root.after(1500, self.reset_extract_ui)

        except Exception as e:
            self.set_output_text(f"❌ ERROR:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()