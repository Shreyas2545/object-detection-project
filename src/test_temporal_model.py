import os
import tempfile
import numpy as np
from temporal_model import TemporalModel
import cv2


def create_static_video(path, n=30, w=160, h=120, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frame = (127 * (np.ones((h, w, 3), dtype='uint8')))
    for _ in range(n):
        writer.write(frame)
    writer.release()


def test_train_and_predict_tmp(tmp_path):
    # create small training set: two AI-like (use sample ai video) and two real static
    ai_video = os.path.join('src', 'ai-video-testing.mp4')
    assert os.path.exists(ai_video)

    real1 = str(tmp_path / 'real1.mp4')
    real2 = str(tmp_path / 'real2.mp4')
    create_static_video(real1)
    create_static_video(real2)

    # Build dataset
    X = []
    y = []
    X.append(TemporalModel.extract_motion_features(ai_video))
    y.append(1)
    X.append(TemporalModel.extract_motion_features(real1))
    y.append(0)
    X.append(TemporalModel.extract_motion_features(real2))
    y.append(0)

    X = np.array(X)
    y = np.array(y)

    tm = TemporalModel()
    tm.fit(X, y)

    # Save and load
    p = str(tmp_path / 'tm.joblib')
    tm.save(p)
    tm2 = TemporalModel()
    tm2.load(p)

    # Predict on ai_video -> expect higher AI prob than on real
    p_ai = tm2.predict_proba(ai_video)
    p_real = tm2.predict_proba(real1)
    assert p_ai >= p_real


if __name__ == '__main__':
    import tempfile
    t = tempfile.TemporaryDirectory()
    test_train_and_predict_tmp(t)