"""
prepare_dataset.py
Розбиває "сирі" зображення (одна папка на клас) на train/val у форматі,
який очікує ImageFolder:

Вхід (raw_dir):
    raw/
        cats/   img1.jpg img2.jpg ...
        dogs/   img1.jpg img2.jpg ...

Вихід (data_dir):
    data/
        train/cats/...  train/dogs/...
        val/cats/...    val/dogs/...

Запуск:
    python src/prepare_dataset.py --raw_dir raw --data_dir data --val_split 0.2
"""

import argparse
import os
import random
import shutil


def prepare(raw_dir, data_dir, val_split, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    if not classes:
        raise ValueError(f"У {raw_dir} не знайдено папок-класів")

    print(f"Знайдено класи: {classes}")

    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)

    for cls in classes:
        src_dir = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_val = max(1, int(len(images) * val_split))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for fname in train_images:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(data_dir, "train", cls, fname))
        for fname in val_images:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(data_dir, "val", cls, fname))

        print(f"  {cls}: {len(train_images)} train, {len(val_images)} val")

    print(f"\nГотово. Датасет підготовлено в: {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Папка з підпапками-класами (сирі зображення)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    prepare(args.raw_dir, args.data_dir, args.val_split)


if __name__ == "__main__":
    main()
