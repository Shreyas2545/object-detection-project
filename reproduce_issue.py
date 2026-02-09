
import sys
import os

# Add src to path if needed (since app.py does it via import)
sys.path.append(os.path.abspath("src"))

try:
    from ai_video_detector import AIVideoDetector
except ImportError:
    # Try local import
    try:
        from src.ai_video_detector import AIVideoDetector
    except ImportError:
        print("Cannot import AIVideoDetector")
        sys.exit(1)

# Mock the detector to force the condition
class MockDetector(AIVideoDetector):
    def __init__(self):
        super().__init__()
        # Ensure we have a mock model so predict_frames_with_model is called
        self.model = True 
        self.model_min_ai_ratio = 0.5 # High enough to force the 'else' block

    def extract_frames(self, video_path):
        # Return dummy frames
        import numpy as np
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * 10, 30, 1.0

    def analyze_temporal_consistency(self, frames):
        return {'score': 10, 'frame_diff_std': 0, 'color_shift_std': 0, 'motion_inconsistency': 0}

    def analyze_frame_artifacts(self, frames):
        return {'score': 10, 'avg_frame_score': 0, 'score_variance': 0}

    def predict_frames_with_model(self, frames):
        # Force a result that has ai_ratio < model_min_ai_ratio (e.g. 0.1 < 0.5)
        return {
            'is_ai': False,
            'confidence': 10.0,
            'ai_frame_ratio': 0.1,  # 10%
            'frame_probs': [0.1] * 10,
            'preds': [0] * 10
        }

if __name__ == "__main__":
    print("Running reproduction script...")
    detector = MockDetector()
    try:
        # We need a dummy file path, doesn't need to exist because we mocked extract_frames
        # But predict checks file size. So we need a real file.
        # Let's create a dummy file.
        with open("dummy_video.mp4", "wb") as f:
            f.write(b"0" * 1024) # 1KB
        
        result = detector.predict("dummy_video.mp4")
        print("Success! Result keys:", result.keys())
        print("Model Confidence:", result['metrics'].get('model_confidence'))
        
    except UnboundLocalError as e:
        print("CAUGHT EXPECTED ERROR:", e)
        sys.exit(1)
    except Exception as e:
        print("Caught unexpected error:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists("dummy_video.mp4"):
            os.remove("dummy_video.mp4")
