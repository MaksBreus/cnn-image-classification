"""
train.py
Тренування CNN (ResNet18, transfer learning) для класифікації зображень.

Очікувана структура датасету:
    data/
        train/
            class1/  img1.jpg  img2.jpg ...
            class2/  ...
        val/
            class1/  ...
            class2/  ...

Запуск:
    python src/train.py --data_dir data --epochs 15 --batch_size 32
"""

import argparse
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir, batch_size, img_size=224):
    """Готує DataLoader'и для train/val з аугментацією для train."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in ["train", "val"]
    }

    dataloaders = {
        split: torch.utils.data.DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
        )
        for split in ["train", "val"]
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def build_model(num_classes, freeze_backbone=True):
    """Завантажує ResNet18 з попередньо натренованими вагами (ImageNet)
    і замінює останній шар під потрібну кількість класів."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_model(model, dataloaders, dataset_sizes, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    # Оптимізуємо тільки новий останній шар (бо backbone заморожений)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    since = time.time()

    for epoch in range(epochs):
        print(f"\nЕпоха {epoch + 1}/{epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase.upper():5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nТренування завершено за {time_elapsed // 60:.0f}хв {time_elapsed % 60:.0f}с")
    print(f"Найкраща Val Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


def plot_history(history, out_path="outputs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss по епохах")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy по епохах")
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Графіки збережено: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    dataloaders, dataset_sizes, class_names = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Класи: {class_names}")
    print(f"Розмір train: {dataset_sizes['train']}, val: {dataset_sizes['val']}")

    model = build_model(num_classes=len(class_names)).to(device)

    model, history = train_model(model, dataloaders, dataset_sizes, device, args.epochs, args.lr)

    # Збереження моделі
    model_path = os.path.join(args.out_dir, "best_model.pth")
    torch.save({"model_state": model.state_dict(), "class_names": class_names}, model_path)
    print(f"Модель збережено: {model_path}")

    # Збереження історії навчання (для звіту)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, os.path.join(args.out_dir, "training_curves.png"))


if __name__ == "__main__":
    main()
