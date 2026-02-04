import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load


class TemporalModel:
    """Simple temporal classifier using motion-derived features (optical flow stats).

    - extract_motion_features(video_path): compute summary stats of frame diffs and optical flow magnitudes
    - fit(X, y) / train_from_dirs(train_ai_dir, train_real_dir)
    - save/load
    - predict_proba(video_path)
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    @staticmethod
    def extract_motion_features(video_path, max_frames=60, resize=(160, 120)):
        """Return a feature vector summarizing motion and temporal changes from a video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_sample = min(max_frames, total)
            if frames_to_sample <= 1:
                cap.release()
                # low information -> return zeros
                return np.zeros(12, dtype=float)

            # sample indices evenly
            idxs = np.linspace(0, total - 1, frames_to_sample, dtype=int)

            prev_gray = None
            diffs = []
            flow_means = []

            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                small = cv2.resize(frame, resize)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    diff = np.mean(cv2.absdiff(prev_gray, gray))
                    diffs.append(diff)

                    # optical flow magnitude
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.4, 2, 9, 2, 5, 1.1, 0)
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    flow_means.append(np.mean(mag))

                prev_gray = gray

            cap.release()

            # summarize
            diffs = np.array(diffs) if diffs else np.zeros(1)
            flow_means = np.array(flow_means) if flow_means else np.zeros(1)

            features = [
                np.mean(diffs), np.std(diffs), np.min(diffs), np.max(diffs),
                np.percentile(diffs, 25), np.percentile(diffs, 75),
                np.mean(flow_means), np.std(flow_means), np.min(flow_means), np.max(flow_means),
                np.percentile(flow_means, 25), np.percentile(flow_means, 75)
            ]

            return np.array(features, dtype=float)

        except Exception:
            # on error, return zero features
            return np.zeros(12, dtype=float)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba_from_features(self, features):
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        # return probability of AI class (assume label 1 is AI)
        # If model was trained with labels [0,1] ordered, check classes_ order
        if hasattr(self.model, 'classes_') and list(self.model.classes_) == [1, 0]:
            # swap if unexpected
            ai_prob = probs[list(self.model.classes_).index(1)]
        else:
            # normally classes_ == [0,1]
            ai_prob = probs[1] if len(probs) > 1 else 0.0
        return float(ai_prob)

    def predict_proba(self, video_path):
        feats = self.extract_motion_features(video_path)
        return self.predict_proba_from_features(feats)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(self.model, path)

    def load(self, path):
        self.model = load(path)

    @staticmethod
    def build_dataset_from_dirs(ai_dir, real_dir):
        X = []
        y = []
        for p in (ai_dir or []):
            if os.path.isdir(p):
                for f in os.listdir(p):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp):
                        X.append(TemporalModel.extract_motion_features(fp))
                        y.append(1)
        for p in (real_dir or []):
            if os.path.isdir(p):
                for f in os.listdir(p):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp):
                        X.append(TemporalModel.extract_motion_features(fp))
                        y.append(0)
        return np.array(X), np.array(y)


if __name__ == '__main__':
    print('TemporalModel utility module. Use train_temporal_model.py to train a model.')
