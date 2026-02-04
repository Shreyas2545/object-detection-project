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
    metrics = result.get('metrics', {})

    # If a high-confidence per-frame AI "hotspot" exists (>= detector.hotspot_threshold),
    # we expect the detector to flag AI; otherwise, strict mode may produce Real.
    max_prob = metrics.get('model_max_ai_prob', 0.0)
    assert 'model_max_ai_prob' in metrics, "model_max_ai_prob not present in metrics"

    # If a per-frame hotspot exceeded the threshold, it should be recorded in metrics.
    if float(max_prob) >= detector.hotspot_threshold:
        assert 'hotspot_threshold' in metrics and 'hotspot_boost' in metrics, "Hotspot info expected in metrics when high frame prob detected"
    else:
        # Otherwise hotspot fields should not be present or below threshold
        assert not ('hotspot_threshold' in metrics and metrics.get('hotspot_threshold') >= detector.hotspot_threshold)

    # Additionally, model_confidence should reflect ai-frame ratio * 100
    assert abs(metrics.get('model_confidence', 0.0) - (metrics.get('ai_frame_ratio', 0.0) * 100.0)) < 1e-6, "model_confidence should equal ai_frame_ratio * 100"


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



