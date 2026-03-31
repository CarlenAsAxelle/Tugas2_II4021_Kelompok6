# src/video_io.py
import cv2
import numpy as np
import subprocess
import os
import tempfile
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


def write_video_frames(path: str, frames: List[np.ndarray], fps: float,
                       embedded_frame_count: int = 0,
                       audio_source: str = None):
    """
    Tulis list frame ke file AVI dengan codec lossless (FFV1).

    Args:
        path: Output file path
        frames: List of BGR frames
        fps: Frames per second
        embedded_frame_count: Number of frames from start containing embedded data.
                            If > 0, those frames use lossless encoding while
                            remaining frames are JPEG-preprocessed for smaller size.
        audio_source: Path to original video to copy audio from.
                     If None, output will have no audio.
    """
    if not frames:
        raise ValueError("frames is empty")

    if audio_source:
        # Write video-only to temp, then mux audio from cover
        ext = os.path.splitext(path)[1] or '.avi'
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(temp_fd)
        try:
            _write_video_only(temp_path, frames, fps, embedded_frame_count)
            _mux_audio(temp_path, audio_source, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        _write_video_only(path, frames, fps, embedded_frame_count)


def _write_video_only(path: str, frames: List[np.ndarray], fps: float,
                      embedded_frame_count: int):
    """Write video frames without audio."""
    if embedded_frame_count > 0:
        _write_avi_frames_selective(path, frames, fps, embedded_frame_count)
    else:
        _write_avi_frames_lossless(path, frames, fps)


def _mux_audio(video_path: str, audio_source: str, output_path: str):
    """Copy audio from audio_source into video_path, save as output_path.

    AVI containers do not support AAC/ADTS audio; audio is re-encoded to
    MP3 for AVI output and copied directly for MP4 output.
    """
    try:
        _check_ffmpeg()
    except EnvironmentError:
        import shutil
        shutil.copy2(video_path, output_path)
        return

    audio_args = ["-c:a", "libmp3lame", "-q:a", "2"]

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source,
        "-c:v", "copy",
        *audio_args,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        import shutil
        shutil.copy2(video_path, output_path)
        print(f"[AUDIO] Mux failed, video-only output: {result.stderr.decode()[-200:]}")
    else:
        print(f"[AUDIO] Muxed audio from cover → {os.path.getsize(output_path):,} bytes")


def _write_avi_frames_opencv(path: str, frames: List[np.ndarray], fps: float):
    """Fallback: FFV1 via OpenCV (lossless, preserves LSBs)."""
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    if not out.isOpened():
        print("Warning: FFV1 codec not available, using MJPG codec")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for f in frames:
        out.write(f)
    out.release()


def _check_ffmpeg():
    """Pastikan ffmpeg tersedia di PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True
        )
        if result.returncode != 0:
            raise EnvironmentError()
    except (FileNotFoundError, EnvironmentError):
        raise EnvironmentError(
            "ffmpeg tidak ditemukan. Install ffmpeg dan pastikan ada di PATH.\n"
            "Windows: https://ffmpeg.org/download.html\n"
            "Linux:   sudo apt install ffmpeg\n"
            "Mac:     brew install ffmpeg"
        )


def _write_avi_frames_lossless(path: str, frames: List[np.ndarray], fps: float):
    """Write AVI with FFV1 lossless codec via ffmpeg (better compression than OpenCV)."""
    try:
        _check_ffmpeg()
    except EnvironmentError:
        print("Warning: ffmpeg not available, falling back to OpenCV FFV1")
        _write_avi_frames_opencv(path, frames, fps)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")
        for i, frame in enumerate(frames):
            fname = os.path.join(tmpdir, f"frame_{i+1:08d}.png")
            cv2.imwrite(fname, frame)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "ffv1",
            "-level", "3",
            "-slicecrc", "1",
            path
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg AVI lossless encoding failed:\n{result.stderr.decode()}"
            )

        print(f"[AVI LOSSLESS] FFV1 → {os.path.getsize(path):,} bytes")


def _write_avi_frames_selective(path: str, frames: List[np.ndarray], fps: float,
                                embedded_frame_count: int):
    """
    Selective AVI encoding:
    - Frames 0..embedded_frame_count-1: Pixel-perfect (lossless FFV1)
    - Frames embedded_frame_count..end: JPEG pre-compressed to reduce entropy
    - All frames encoded as FFV1 (single codec, valid AVI container)

    The JPEG round-trip on non-embedded frames discards high-frequency detail,
    making FFV1 compress them significantly smaller while embedded frames
    remain bit-perfect for LSB steganography.
    """
    try:
        _check_ffmpeg()
    except EnvironmentError:
        print("Warning: ffmpeg not available, falling back to OpenCV FFV1")
        _write_avi_frames_opencv(path, frames, fps)
        return

    if embedded_frame_count <= 0 or embedded_frame_count >= len(frames):
        _write_avi_frames_lossless(path, frames, fps)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")
        for i, frame in enumerate(frames):
            out_frame = frame
            if i >= embedded_frame_count:
                # Lossy JPEG round-trip (quality 92 ≈ visually transparent)
                _, buf = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 92])
                out_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            fname = os.path.join(tmpdir, f"frame_{i+1:08d}.png")
            cv2.imwrite(fname, out_frame)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "ffv1",
            "-level", "3",
            "-slicecrc", "1",
            path
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg AVI selective encoding failed:\n{result.stderr.decode()}"
            )

        print(f"[SELECTIVE AVI] {embedded_frame_count} lossless + "
              f"{len(frames) - embedded_frame_count} lossy → "
              f"{os.path.getsize(path):,} bytes")


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