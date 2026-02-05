"""
Training script for AI Image Detector (binary classifier: Real vs AI-generated)

Usage:
  python src/train_ai_detector.py --data_dir data/ai_detector --epochs 10 --batch_size 32

Folder layout expected:
  data/ai_detector/train/real/
  data/ai_detector/train/ai/
  data/ai_detector/val/real/
  data/ai_detector/val/ai/

Saves checkpoint to: checkpoints/ai_detector.pth
"""
import os
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def _temperature_scale(logits, labels, max_iters=200, lr=0.01):
    """Fit a scalar temperature parameter for logits to calibrate confidence."""
    import torch
    T = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    optimizer = torch.optim.LBFGS([T], lr=lr)
    nll_criterion = nn.CrossEntropyLoss()

    logits = logits.clone()
    labels = labels.clone()

    def _eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits / T, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(_eval)
    except Exception:
        pass

    return float(T.item())


def train(data_dir, epochs=10, batch_size=32, lr=1e-4, save_path='checkpoints/ai_detector.pth', calibrate=True, focal_gamma=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.1)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Starting training: epochs={epochs}, batch_size={batch_size}, train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Compute class weights to handle imbalance
    counts = {}
    for _, label in train_dataset.imgs:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    class_weights = [total / counts.get(i, 1) for i in range(2)]
    class_weights = torch.tensor(class_weights).float().to(device)

    if focal_gamma > 0.0:
        # Implement focal loss if requested (per-sample reduction)
        def focal_loss(inputs, targets, gamma=focal_gamma):
            ce = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            p = torch.softmax(inputs, dim=1)
            pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = ((1 - pt) ** gamma) * ce
            return loss.mean()
        criterion = focal_loss
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    best_metric = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if callable(criterion):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total * 100

        # Validation (collect logits for calibration)
        model.eval()
        val_correct = 0
        val_total = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                logits = outputs.detach().cpu()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
                all_logits.append(logits)
                all_labels.append(labels.cpu())

        val_acc = val_correct / val_total * 100
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Optionally temperature scale on validation logits at end of training run
        temp = 1.0
        if calibrate and epoch == epochs:
            try:
                temp = _temperature_scale(all_logits, all_labels)
                print(f"Calibrated temperature: {temp:.3f}")
            except Exception:
                temp = 1.0

        # Compute calibrated probabilities and PR metrics
        probs = torch.softmax(all_logits / temp, dim=1).numpy()
        preds = probs.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels.numpy(), preds, average='binary')
        try:
            auc = roc_auc_score(all_labels.numpy(), probs[:, 1])
        except Exception:
            auc = 0.0

        metric_score = f1

        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.2f}% - val_acc: {val_acc:.2f}% - val_f1: {f1:.3f} - val_auc: {auc:.3f}")

        # Save best model by F1
        if metric_score > best_metric:
            best_metric = metric_score
            best_state = model.state_dict()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f"Saved improved model to {save_path} (val_f1 {f1:.3f})")

        scheduler.step()

    # Final calibration step on best model
    if best_state is not None and calibrate:
        model.load_state_dict(best_state)
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                all_logits.append(outputs.detach().cpu())
                all_labels.append(labels.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        try:
            temp = _temperature_scale(all_logits, all_labels)
            print(f"Final calibrated temperature: {temp:.3f}")
            # save temperature alongside model
            torch.save({'state_dict': best_state, 'temperature': temp}, save_path)
        except Exception:
            torch.save({'state_dict': best_state, 'temperature': 1.0}, save_path)

    print(f"Training complete. Best val F1: {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ai_detector', help='data directory with train/val subdirs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', default='checkpoints/ai_detector.pth')
    parser.add_argument('--calibrate', action='store_true', help='Run temperature scaling calibration on validation set')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='Use focal loss with this gamma (>0 for focal)')

    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path, calibrate=args.calibrate, focal_gamma=args.focal_gamma)
