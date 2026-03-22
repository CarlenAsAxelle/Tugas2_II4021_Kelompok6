# tests/test_video_io.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import os
import pytest
import numpy as np
from src.video_io import (
    read_video_frames, write_video_frames, 
    mse_frame, psnr_frame, mse_psnr_video,
    color_histogram_frame, color_histogram_video
)

SAMPLE_VIDEO = "samples/sample_video.avi"

@pytest.fixture
def sample_frames():
    """
    Load 2 frame pertama dari sample_video.avi (bukan random).
    """
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip(f"Sample video tidak ada: {SAMPLE_VIDEO}")
    
    frames, fps = read_video_frames(SAMPLE_VIDEO)
    assert len(frames) >= 2, "Video harus punya minimal 2 frame"
    return frames[:2], fps  # ambil 2 frame pertama saja

def test_read_write_video_cycle(sample_frames, tmp_path):
    frames, fps = sample_frames
    temp_avi = tmp_path / "test_cycle.avi"
    temp_read_avi = tmp_path / "test_read_result.avi"  # save hasil baca
    
    print(f"Original frames shape: {len(frames)} x {frames[0].shape}")
    
    # WRITE
    write_video_frames(str(temp_avi), frames, fps)
    print(f"✓ Wrote {temp_avi} ({os.path.getsize(str(temp_avi))//1000} KB)")
    
    # READ  
    read_frames, read_fps = read_video_frames(str(temp_avi))
    write_video_frames(str(temp_read_avi), read_frames, read_fps)
    print(f"✓ Read {len(read_frames)} frames @ {read_fps:.1f} FPS")
    
    # METRICS
    mse_list, psnr_list, mse_avg, psnr_avg = mse_psnr_video(frames, read_frames)
    print(f"MSE: avg={mse_avg:.2f}, max={max(mse_list):.2f}")
    print(f"PSNR: avg={psnr_avg:.2f} dB, max={max(psnr_list):.2f} dB")
    
    # SAVE VIDEO KE ROOT BIAR MUDA LIHAT
    root_output = "tests_output/test_output.avi"
    write_video_frames(root_output, read_frames, read_fps)
    print(f"Saved test video: {root_output} (buka dengan VLC/Media Player)")
    
    # ASSERT (toleransi codec)
    assert len(read_frames) == len(frames)
    assert all(f.shape == rf.shape for f, rf in zip(frames, read_frames))
    assert mse_avg < 10000  # toleransi codec Windows

def test_mse_psnr_functions(sample_frames):
    """
    Test MSE/PSNR di level frame dan video. [file:1]
    """
    frames, _ = sample_frames
    frame1, frame2 = frames
    
    # MSE/PSNR frame
    mse_val = mse_frame(frame1, frame2)
    psnr_val = psnr_frame(frame1, frame2)
    assert 0 <= mse_val <= 65025  # (255^2)
    assert 0 <= psnr_val <= np.inf
    
    # Test identik frame (PSNR inf)
    mse_identik = mse_frame(frame1, frame1)
    psnr_identik = psnr_frame(frame1, frame1)
    assert mse_identik == 0
    assert np.isinf(psnr_identik)
    
    # MSE/PSNR video
    mse_list, psnr_list, mse_avg, psnr_avg = mse_psnr_video([frame1], [frame2])
    assert len(mse_list) == 1
    assert mse_avg == mse_val

def test_histogram_functions(sample_frames):
    """
    Test histogram RGB per frame dan video. [file:1]
    """
    frames, _ = sample_frames
    frame = frames[0]
    
    # histogram satu frame
    hist_b, hist_g, hist_r = color_histogram_frame(frame)
    assert hist_b.shape == (256, 1)
    assert np.sum(hist_b) == frame.shape[0] * frame.shape[1]  # total pixel
    
    # histogram video (rata-rata)
    hist_b_avg, hist_g_avg, hist_r_avg = color_histogram_video([frame])
    assert np.array_equal(hist_b_avg, hist_b)  # single frame = rata-rata itu sendiri

def test_error_handling(sample_frames):
    """
    Test error cases.
    """
    frames, _ = sample_frames
    frame1 = frames[0]
    
    # MSE ukuran berbeda
    frame_wrong = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Ukuran frame berbeda"):
        mse_frame(frame1, frame_wrong)
    
    # PSNR inf
    assert np.isinf(psnr_frame(frame1, frame1))
    
    # write frames kosong
    with pytest.raises(ValueError, match="frames is empty"):
        write_video_frames("dummy.avi", [], 30.0)

def test_real_avi_file(sample_video_path):
    """
    Test dengan file AVI asli (skip kalau file tidak ada).
    """
    if not os.path.exists(sample_video_path):
        pytest.skip(f"Sample video tidak ada: {sample_video_path}")
    
    frames, fps = read_video_frames(sample_video_path)
    assert len(frames) > 0
    assert fps > 0
    assert frames[0].ndim == 3  # HxWx3
    
    print(f"Loaded {len(frames)} frames at {fps:.1f} FPS from {sample_video_path}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])