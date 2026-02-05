import os
import sys
sys.path.append('.')
from ai_video_detector import AIVideoDetector
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def evaluate(data_dir='data/ai_videos/val'):
    D = AIVideoDetector()
    y_true = []
    y_pred = []
    files = []
    for cls, label in [('ai',1), ('real',0)]:
        folder = Path(data_dir) / cls
        if not folder.exists():
            continue
        for f in folder.glob('*.mp4'):
            print('Testing', f)
            res = D.predict(str(f))
            pred = 1 if res['is_ai_generated'] else 0
            y_true.append(label)
            y_pred.append(pred)
            files.append((str(f), label, pred, res.get('metrics',{})))

    if not y_true:
        print('No validation videos found. Prepare data/ai_videos/val/...')
        return

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    print('\nResults:')
    print('  Precision:', precision)
    print('  Recall:', recall)
    print('  F1:', f1)
    print('  Confusion matrix:\n', cm)
    return files, precision, recall, f1, cm

if __name__ == '__main__':
    evaluate()
