import os
import tempfile
import cv2
import numpy as np
from ai_video_detector import AIVideoDetector


def test_hotspot_detection_on_sample_video():
    """Ensure the provided AI sample triggers hotspot-based AI labeling"""
    detector = AIVideoDetector()
    sample_video = os.path.join('src', 'ai-video-testing.mp4')
    assert os.path.exists(sample_video), "Sample video missing"

    result = detector.predict(sample_video)
    # Expect hotspot to trigger causing is_ai_generated True
    assert bool(result['is_ai_generated']) is True, f"Expected AI-generated, got {result['label']}"

    # Check that model_max_ai_prob exists and is above hotspot threshold
    metrics = result.get('metrics', {})
    assert 'model_max_ai_prob' in metrics, "model_max_ai_prob not present in metrics"
    assert metrics['model_max_ai_prob'] >= detector.hotspot_threshold


def test_synthetic_static_video_classifies_real(tmp_path):
    """Create a synthetic static video (no motion, no artifacts) and expect Real"""
    # Create a temporary video file with 30 identical frames (simple static scene)
    video_path = tmp_path / "static_test.mp4"
    width, height = 320, 240
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Make 30 frames of a plain gray image
    frame = (127 * (np.ones((height, width, 3), dtype='uint8')))
    for _ in range(30):
        writer.write(frame)
    writer.release()

    detector = AIVideoDetector()
    result = detector.predict(str(video_path))

    # Static video should not be classified as AI-generated
    assert bool(result['is_ai_generated']) is False, f"Expected Real, got {result['label']}"



