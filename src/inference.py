"""
inference.py
Класифікація ОДНОГО нового зображення натренованою моделлю.

Запуск:
    python src/inference.py --image path/to/photo.jpg --model_path outputs/best_model.pth
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names


def predict(image_path, model, class_names, device, img_size=224):
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, 0)

    print(f"\nЗображення: {image_path}")
    print(f"Передбачений клас: {class_names[top_idx]} (впевненість: {top_prob:.2%})\n")
    print("Ймовірності по всіх класах:")
    for name, p in sorted(zip(class_names, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {name:20s} {p:.2%}")

    return class_names[top_idx], top_prob.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Шлях до зображення")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model_path, device)
    predict(args.image, model, class_names, device)


if __name__ == "__main__":
    main()
