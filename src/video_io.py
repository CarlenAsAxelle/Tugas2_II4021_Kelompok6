# src/video_io.py
import cv2
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor


def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Baca semua frame dari video AVI.
    Return: (list_frame_BGR_uint8, fps)
    
    Note: Reads frames incrementally into a list to avoid massive memory allocation.
    For large videos, this is more efficient than pre-allocating a giant buffer.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Read frames incrementally - this avoids OOM errors on large videos
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
    Tulis list frame ke file AVI dengan codec lossless (FFv1).
    """
    if not frames:
        raise ValueError("frames is empty")

    h, w, _ = frames[0].shape
    # Use FFv1 which is lossless - essential for preserving LSBs in steganography
    fourcc   = cv2.VideoWriter_fourcc(*"FFV1")
    out      = cv2.VideoWriter(path, fourcc, fps, (w, h))

    if not out.isOpened():
        # FFv1 might not be available, try with MJPEG as fallback (still better than XVID for LSBs)
        print("Warning: FFv1 codec not available, using MJPG (lossless-ish) codec")
        fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
        out      = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for f in frames:
        out.write(f)

    out.release()


# ─── METRICS ──────────────────────────────────────────────────────────────────

def _mse_psnr_pair(args: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
    """Compute MSE + PSNR for one (cover, stego) pair — used in parallel map."""
    ref, stego = args
    diff    = ref.astype(np.float32) - stego.astype(np.float32)
    mse_val = float(np.mean(diff * diff))
    if mse_val == 0:
        return mse_val, float('inf')
    psnr_val = float(10.0 * np.log10(65025.0 / mse_val))  # 255^2 = 65025
    return mse_val, psnr_val


def mse_frame(ref: np.ndarray, stego: np.ndarray) -> float:
    """
    Hitung MSE antara dua frame (BGR, uint8, ukuran sama).
    """
    if ref.shape != stego.shape:
        raise ValueError("Ukuran frame berbeda")
    diff = ref.astype(np.float32) - stego.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr_frame(ref: np.ndarray, stego: np.ndarray) -> float:
    """
    Hitung PSNR (dalam dB) untuk dua frame.
    """
    mse_val = mse_frame(ref, stego)
    if mse_val == 0:
        return float('inf')
    return float(10.0 * np.log10(65025.0 / mse_val))


def mse_psnr_video(
    cover_frames: List[np.ndarray],
    stego_frames: List[np.ndarray]
) -> Tuple[List[float], List[float], float, float]:
    """
    Hitung MSE & PSNR per frame dan rata-rata untuk dua video (list frame).
    Menggunakan thread pool untuk mempercepat komputasi paralel.
    """
    if len(cover_frames) != len(stego_frames):
        raise ValueError("Jumlah frame video berbeda")

    pairs = list(zip(cover_frames, stego_frames))

    # Parallel computation across frames using I/O-friendly thread pool
    workers = min(8, len(pairs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_mse_psnr_pair, pairs))

    mse_list  = [r[0] for r in results]
    psnr_list = [r[1] for r in results]
    mse_avg   = float(np.mean(mse_list))
    psnr_avg  = float(np.mean([p for p in psnr_list if not np.isinf(p)] or [float('inf')]))

    return mse_list, psnr_list, mse_avg, psnr_avg


# ─── HISTOGRAMS ───────────────────────────────────────────────────────────────

def color_histogram_frame(
    frame: np.ndarray, bins: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung histogram warna B, G, R untuk satu frame.
    Return: (hist_b, hist_g, hist_r) dengan shape (bins,1).
    """
    channels = cv2.split(frame)
    hist_b   = cv2.calcHist([channels[0]], [0], None, [bins], [0, 256])
    hist_g   = cv2.calcHist([channels[1]], [0], None, [bins], [0, 256])
    hist_r   = cv2.calcHist([channels[2]], [0], None, [bins], [0, 256])
    return hist_b, hist_g, hist_r


def _hist_frame_numpy(frame: np.ndarray, bins: int = 256):
    """Fast per-frame histogram via NumPy (avoids OpenCV call overhead in loops)."""
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    hb = np.bincount(b.ravel(), minlength=bins).reshape(bins, 1).astype(np.float32)
    hg = np.bincount(g.ravel(), minlength=bins).reshape(bins, 1).astype(np.float32)
    hr = np.bincount(r.ravel(), minlength=bins).reshape(bins, 1).astype(np.float32)
    return hb, hg, hr


def color_histogram_video(
    frames: List[np.ndarray], bins: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung histogram rata-rata B, G, R untuk seluruh frame video.
    Menggunakan NumPy bincount (lebih cepat dari cv2.calcHist dalam loop).
    """
    if not frames:
        raise ValueError("frames is empty")

    n             = len(frames)
    hist_b_sum    = np.zeros((bins, 1), dtype=np.float32)
    hist_g_sum    = np.zeros((bins, 1), dtype=np.float32)
    hist_r_sum    = np.zeros((bins, 1), dtype=np.float32)

    for f in frames:
        hb, hg, hr = _hist_frame_numpy(f, bins=bins)
        hist_b_sum += hb
        hist_g_sum += hg
        hist_r_sum += hr

    return hist_b_sum / n, hist_g_sum / n, hist_r_sum / n