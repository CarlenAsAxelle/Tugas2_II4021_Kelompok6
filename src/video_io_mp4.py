# src/video_io_mp4.py
# MP4-only video I/O. AVI is handled by video_io.py.
import cv2
import numpy as np
import subprocess
import os
import tempfile
from typing import List, Tuple


# ─── FFMPEG CHECK ─────────────────────────────────────────────────────────────

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


# ─── READ / WRITE PUBLIC API ─────────────────────────────────────────────────

def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Baca semua frame dari video MP4 via ffmpeg (lossless PNG roundtrip).
    """
    return _read_mp4_frames_lossless(path)


def write_video_frames(path: str, frames: List[np.ndarray], fps: float,
                       mp4_crf: int = 0, embedded_frame_count: int = 0,
                       audio_source: str = None):
    """
    Tulis list frame ke file MP4.

    Args:
        path: Output file path (.mp4)
        frames: List of BGR frames
        fps: Frames per second
        mp4_crf: CRF value for MP4 lossy encoding (ignored if 0)
        embedded_frame_count: Number of frames from start containing embedded data.
                            If > 0, those frames use lossless encoding while
                            remaining frames use lossy (CRF 23).
        audio_source: Path to original video to copy audio from.
                     If None, output will have no audio.
    """
    if not frames:
        raise ValueError("frames is empty")

    if audio_source:
        # Write video-only to temp, then mux audio from cover
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        try:
            _write_video_only(temp_path, frames, fps, mp4_crf, embedded_frame_count)
            _mux_audio(temp_path, audio_source, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        _write_video_only(path, frames, fps, mp4_crf, embedded_frame_count)


def _write_video_only(path: str, frames: List[np.ndarray], fps: float,
                      mp4_crf: int, embedded_frame_count: int):
    """Write video frames without audio."""
    if embedded_frame_count > 0:
        _write_mp4_frames_selective(path, frames, fps, embedded_frame_count)
    else:
        _write_mp4_frames_lossless(path, frames, fps, crf=mp4_crf)


def _mux_audio(video_path: str, audio_source: str, output_path: str):
    """Copy audio from audio_source into video_path, save as output_path."""
    try:
        _check_ffmpeg()
    except EnvironmentError:
        import shutil
        shutil.copy2(video_path, output_path)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,       # video (no audio)
        "-i", audio_source,     # source of audio
        "-c:v", "copy",         # don't re-encode video
        "-c:a", "copy",         # don't re-encode audio
        "-map", "0:v:0",        # take video from first input
        "-map", "1:a?",         # take audio from second input (optional)
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # If muxing fails (e.g., no audio in source), just copy video-only
        import shutil
        shutil.copy2(video_path, output_path)
        print(f"[AUDIO] Mux failed, video-only output: {result.stderr.decode()[-200:]}")
    else:
        print(f"[AUDIO] Muxed audio from cover → {os.path.getsize(output_path):,} bytes")


# ─── MP4 INTERNAL (via ffmpeg, lossless PNG roundtrip) ────────────────────────

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
    Gunakan libx264rgb untuk menghindari konversi warna yang merusak LSB.
    """
    _check_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")
        for i, frame in enumerate(frames):
            fname = os.path.join(tmpdir, f"frame_{i+1:08d}.png")
            cv2.imwrite(fname, frame)

        if crf == 0:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264rgb",
                "-preset", "slow",
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
                "-pix_fmt", "yuv420p",
                path
            ]

        print(f"\n[FFMPEG] Command: {' '.join(cmd)}")
        print(f"[FFMPEG] Frames: {len(frames)}, FPS: {fps}")

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr_msg = result.stderr.decode()
            print(f"\n[DEBUG] ffmpeg failed with return code {result.returncode}")
            print(f"[DEBUG] stderr: {stderr_msg[-500:]}")
            raise RuntimeError(
                f"ffmpeg gagal menulis MP4:\n{stderr_msg}"
            )

        output_size = os.path.getsize(path)
        print(f"[FFMPEG] Output: {output_size:,} bytes")


def _write_mp4_frames_selective(path: str, frames: List[np.ndarray], fps: float,
                                embedded_frame_count: int):
    """
    Tulis MP4 dengan selective lossless/lossy encoding untuk kompresi maksimal.

    Strategy:
    - Frame 0 to embedded_frame_count-1: Lossless (CRF 0) lindungi LSB data
    - Frame embedded_frame_count onwards: Lossy (CRF 23) compress signifikan
    - Concat dengan -c copy (instant, no re-encode)
    """
    _check_ffmpeg()

    if embedded_frame_count <= 0 or embedded_frame_count >= len(frames):
        _write_mp4_frames_lossless(path, frames, fps, crf=0)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        lossless_file = os.path.join(tmpdir, "lossless.mp4")
        lossy_file = os.path.join(tmpdir, "lossy.mp4")

        # Part 1: Lossless (frames dengan embedded data)
        lossless_frames = frames[:embedded_frame_count]
        _write_mp4_frames_lossless(lossless_file, lossless_frames, fps, crf=0)

        # Part 2: Lossy (frames tanpa embedded data)
        lossy_frames = frames[embedded_frame_count:]
        _write_mp4_frames_simple_lossy(lossy_file, lossy_frames, fps, crf=23)

        # Concat dengan -c copy (instant, tidak re-encode)
        _concatenate_mp4_videos(lossless_file, lossy_file, path)

        print(f"[SELECTIVE MP4] {embedded_frame_count} lossless + "
              f"{len(lossy_frames)} lossy → {os.path.getsize(path):,} bytes")


def _write_mp4_frames_simple_lossy(path: str, frames: List[np.ndarray],
                                   fps: float, crf: int = 23):
    """
    Tulis MP4 dengan lossy encoding (CRF configurable).
    Untuk frames yang tidak contain embedded data.
    """
    _check_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%08d.png")
        for i, frame in enumerate(frames):
            fname = os.path.join(tmpdir, f"frame_{i+1:08d}.png")
            cv2.imwrite(fname, frame)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            path
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg lossy encoding failed:\n{result.stderr.decode()}"
            )


def _concatenate_mp4_videos(part1: str, part2: str, output: str):
    """
    Concatenate two MP4 files using ffmpeg concat demuxer (instant, no re-encode).
    """
    _check_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir:
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, 'w') as f:
            f.write(f"file '{os.path.abspath(part1)}'\n")
            f.write(f"file '{os.path.abspath(part2)}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed:\n{result.stderr.decode()}"
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

    return mse_list, psnr_list, mse_avg, psnr_avg