# src/video_io.py
import cv2
import numpy as np
from typing import List, Tuple

def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Baca semua frame dari video AVI.
    Return: (list_frame_BGR_uint8, fps)
    """
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

def write_video_frames(path: str, frames: List[np.ndarray], fps: float):
    """
    Tulis list frame ke file AVI.
    """
    if not frames:
        raise ValueError("frames is empty")

    h, w, c = frames[0].shape
    # FFV1: lossless video codec (ideal untuk test) [web:38]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for f in frames:
        out.write(f)

    out.release()


def mse_frame(ref: np.ndarray, stego: np.ndarray) -> float:
    """
    Hitung MSE antara dua frame (BGR, uint8, ukuran sama). [web:35][file:1]
    """
    if ref.shape != stego.shape:
        raise ValueError("Ukuran frame berbeda")

    diff = ref.astype(np.float32) - stego.astype(np.float32)
    mse_val = np.mean(diff ** 2)
    return float(mse_val)

def psnr_frame(ref: np.ndarray, stego: np.ndarray) -> float:
    """
    Hitung PSNR (dalam dB) untuk dua frame. [web:32][web:38][file:1]
    """
    mse_val = mse_frame(ref, stego)
    if mse_val == 0:
        return float('inf')

    max_i = 255.0  # intensitas maksimum piksel 8-bit [file:1]
    psnr_val = 10 * np.log10((max_i ** 2) / mse_val)
    return float(psnr_val)

def mse_psnr_video(cover_frames: List[np.ndarray],
                   stego_frames: List[np.ndarray]) -> Tuple[List[float], List[float], float, float]:
    """
    Hitung MSE & PSNR per frame dan rata-rata untuk dua video (list frame).
    Diasumsikan jumlah frame sama. [file:1]
    """
    if len(cover_frames) != len(stego_frames):
        raise ValueError("Jumlah frame video berbeda")

    mse_list = []
    psnr_list = []

    for c, s in zip(cover_frames, stego_frames):
        mse_val = mse_frame(c, s)
        psnr_val = psnr_frame(c, s)
        mse_list.append(mse_val)
        psnr_list.append(psnr_val)

    mse_avg = float(np.mean(mse_list))
    psnr_avg = float(np.mean(psnr_list))

    return mse_list, psnr_list, mse_avg, psnr_avg

def color_histogram_frame(frame: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung histogram warna B, G, R untuk satu frame.
    Return: (hist_b, hist_g, hist_r) dengan shape (bins,1). [web:33][web:36][web:39][file:1]
    """
    channels = cv2.split(frame)  # B, G, R
    hist_b = cv2.calcHist([channels[0]], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([channels[1]], [0], None, [bins], [0, 256])
    hist_r = cv2.calcHist([channels[2]], [0], None, [bins], [0, 256])
    return hist_b, hist_g, hist_r

def color_histogram_video(frames: List[np.ndarray], bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung histogram rata-rata B, G, R untuk seluruh frame video. [file:1]
    """
    if not frames:
        raise ValueError("frames is empty")

    hist_b_sum = np.zeros((bins, 1), dtype=np.float32)
    hist_g_sum = np.zeros((bins, 1), dtype=np.float32)
    hist_r_sum = np.zeros((bins, 1), dtype=np.float32)

    for f in frames:
        hb, hg, hr = color_histogram_frame(f, bins=bins)
        hist_b_sum += hb
        hist_g_sum += hg
        hist_r_sum += hr

    n = len(frames)
    return hist_b_sum / n, hist_g_sum / n, hist_r_sum / n
