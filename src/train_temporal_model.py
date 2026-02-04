import argparse
import os
from temporal_model import TemporalModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def main(data_dir='data/ai_videos', save_path='checkpoints/temporal_model.joblib'):
    ai_dir = os.path.join(data_dir, 'train', 'ai')
    real_dir = os.path.join(data_dir, 'train', 'real')

    print(f"Looking for videos in:\n  AI: {ai_dir}\n  Real: {real_dir}")

    X, y = TemporalModel.build_dataset_from_dirs([ai_dir], [real_dir])
    if len(y) == 0:
        print('No video files found. Please prepare dataset: data/ai_videos/train/ai and /real')
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = TemporalModel()
    model.fit(X_train, y_train)
    preds = model.model.predict(X_val)
    probs = model.model.predict_proba(X_val)[:, 1]

    print('\nClassification report:')
    print(classification_report(y_val, preds))
    try:
        print('AUC:', roc_auc_score(y_val, probs))
    except Exception:
        pass

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved temporal model to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ai_videos', help='directory with train/ai and train/real subfolders')
    parser.add_argument('--save_path', default='checkpoints/temporal_model.joblib')
    args = parser.parse_args()

    main(data_dir=args.data_dir, save_path=args.save_path)
