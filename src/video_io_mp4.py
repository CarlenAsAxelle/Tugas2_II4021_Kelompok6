# src/video_io_mp4.py
import cv2
import numpy as np
import subprocess
import os
import tempfile
from typing import List, Tuple


# ─── FORMAT DETECTION ─────────────────────────────────────────────────────────

def get_format(path: str) -> str:
    """Return 'avi' or 'mp4' based on file extension (lowercase)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mp4', '.m4v'):
        return 'mp4'
    elif ext in ('.avi',):
        return 'avi'
    else:
        raise ValueError(f"Unsupported format: {ext}. Only .avi and .mp4 are supported.")


# ─── AVI I/O (original, unchanged) ───────────────────────────────────────────

def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Baca semua frame dari video AVI atau MP4.
    """
    fmt = get_format(path)
    if fmt == 'mp4':
        return _read_mp4_frames_lossless(path)
    else:
        return _read_avi_frames(path)


def write_video_frames(path: str, frames: List[np.ndarray], fps: float,
                       mp4_crf: int = 0):
    """
    Tulis list frame ke file AVI atau MP4.
    """
    if not frames:
        raise ValueError("frames is empty")

    fmt = get_format(path)
    if fmt == 'mp4':
        _write_mp4_frames_lossless(path, frames, fps, crf=mp4_crf)
    else:
        _write_avi_frames(path, frames, fps)


# ─── AVI INTERNAL ─────────────────────────────────────────────────────────────

def _read_avi_frames(path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def _write_avi_frames(path: str, frames: List[np.ndarray], fps: float):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


# ─── MP4 INTERNAL (via ffmpeg, lossless PNG roundtrip) ────────────────────────

def _check_ffmpeg():
    """Pastikan ffmpeg tersedia di PATH."""
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True
    )
    if result.returncode != 0:
        raise EnvironmentError(
            "ffmpeg tidak ditemukan. Install ffmpeg dan pastikan ada di PATH.\n"
            "Windows: https://ffmpeg.org/download.html\n"
            "Linux:   sudo apt install ffmpeg\n"
            "Mac:     brew install ffmpeg"
        )


def _read_mp4_frames_lossless(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Ekstrak frame dari MP4 ke PNG lossless via ffmpeg, lalu baca dengan OpenCV.
    """
    _check_ffmpeg()

    # Ambil FPS dulu via OpenCV
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")

        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-vsync", "0",
            # Hapus pix_fmt bgr24 di sini, biarkan ffmpeg detect native png
            # "-pix_fmt", "bgr24", 
            "-f", "image2",
            "-vcodec", "png",
            frame_pattern
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg gagal membaca MP4:\n{result.stderr.decode()}"
            )

        png_files = sorted([
            f for f in os.listdir(tmpdir) if f.endswith('.png')
        ])

        if not png_files:
            raise RuntimeError("Tidak ada frame yang berhasil diekstrak dari MP4.")

        frames = []
        for fname in png_files:
            img = cv2.imread(os.path.join(tmpdir, fname))
            if img is not None:
                frames.append(img)

    return frames, fps


def _write_mp4_frames_lossless(path: str, frames: List[np.ndarray],
                                fps: float, crf: int = 0):
    """
    Tulis frames ke MP4.
    Fix: Gunakan libx264rgb untuk menghindari konversi warna yang merusak LSB.
    """
    _check_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")
        for i, frame in enumerate(frames):
            fname = os.path.join(tmpdir, f"frame_{i+1:08d}.png")
            cv2.imwrite(fname, frame)

        if crf == 0:
            # FIX: Gunakan libx264rgb, yang didesain khusus untuk RGB lossless.
            # Hapus -pix_fmt gbrp manual, biarkan encoder pilih best match (bgr24/rgb24/gbrp)
            # yang kompatibel dengan PNG input.
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264rgb", # Special encoder for RGB
                "-preset", "ultrafast",
                "-crf", "0",
                path
            ]
        else:
            # Lossy mode
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p", # Standard compatibility
                path
            ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg gagal menulis MP4:\n{result.stderr.decode()}"
            )


# ─── METRICS ──────────────────────────────────────────────────────────────────

def mse_frame(ref: np.ndarray, stego: np.ndarray) -> float:
    diff = ref.astype(np.float64) - stego.astype(np.float64)
    return np.mean(diff ** 2)

def psnr_frame(mse: float) -> float:
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)

def mse_psnr_video(original_frames, stego_frames):
    mse_list = []
    
    # Compare only common frames
    n = min(len(original_frames), len(stego_frames))
    
    for i in range(n):
        mse = mse_frame(original_frames[i], stego_frames[i])
        mse_list.append(mse)

    mse_avg = np.mean(mse_list) if mse_list else 0
    psnr_avg = psnr_frame(mse_avg)
    psnr_list = [psnr_frame(m) for m in mse_list]

    return mse_avg, psnr_list, mse_avg, psnr_avg