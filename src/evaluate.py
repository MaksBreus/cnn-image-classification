"""
evaluate.py
Оцінка натренованої моделі на валідаційній вибірці:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix (збережена як зображення для звіту)

Запуск:
    python src/evaluate.py --data_dir data --model_path outputs/best_model.pth
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from torchvision import datasets, models, transforms
import torch.nn as nn


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names


def get_val_loader(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def evaluate(model, loader, device, class_names):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1, report, cm


def plot_confusion_matrix(cm, class_names, out_path="outputs/confusion_matrix.png"):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Передбачений клас")
    plt.ylabel("Справжній клас")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix збережено: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pth")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names = load_model(args.model_path, device)
    loader = get_val_loader(args.data_dir)

    acc, f1, report, cm = evaluate(model, loader, device, class_names)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}\n")
    print("Детальний звіт по класах:")
    print(report)

    # Збереження текстового звіту
    with open(os.path.join(args.out_dir, "eval_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1-score (weighted): {f1:.4f}\n\n{report}")

    plot_confusion_matrix(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
