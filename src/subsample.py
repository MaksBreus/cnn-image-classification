"""
subsample.py
Бере датасет Intel Image Classification (структура seg_train/seg_train/<клас>,
seg_test/seg_test/<клас>) і створює зменшену підмножину у форматі data/train, data/val
— готову для train.py.

Приклад структури на вході (те, що вийшло після розпакування archive.zip):
    archive/
        seg_train/seg_train/buildings/ forest/ glacier/ mountain/ sea/ street/
        seg_test/seg_test/buildings/ forest/ glacier/ mountain/ sea/ street/

Запуск (з кореня проєкту, поруч з розпакованою папкою archive):
    python src/subsample.py --archive_dir archive --data_dir data \
        --train_per_class 400 --val_per_class 100
"""

import argparse
import os
import random
import shutil


def copy_subset(src_dir, dst_dir, n, seed=42):
    random.seed(seed)
    os.makedirs(dst_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    chosen = images[:n]

    for fname in chosen:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    return len(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir", type=str, default="archive",
                         help="Папка з розпакованим датасетом (з seg_train, seg_test)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_per_class", type=int, default=400)
    parser.add_argument("--val_per_class", type=int, default=100)
    args = parser.parse_args()

    train_src_root = os.path.join(args.archive_dir, "seg_train", "seg_train")
    val_src_root = os.path.join(args.archive_dir, "seg_test", "seg_test")

    if not os.path.isdir(train_src_root):
        raise FileNotFoundError(
            f"Не знайдено {train_src_root}. Перевір шлях --archive_dir "
            f"(має вказувати на папку, де лежать seg_train/seg_test/seg_pred)."
        )

    classes = sorted(d for d in os.listdir(train_src_root)
                      if os.path.isdir(os.path.join(train_src_root, d)))
    print(f"Знайдено класи: {classes}\n")

    for cls in classes:
        n_train = copy_subset(
            os.path.join(train_src_root, cls),
            os.path.join(args.data_dir, "train", cls),
            args.train_per_class,
        )
        n_val = copy_subset(
            os.path.join(val_src_root, cls),
            os.path.join(args.data_dir, "val", cls),
            args.val_per_class,
        )
        print(f"{cls:12s}: train={n_train}, val={n_val}")

    print(f"\nГотово. Зменшений датасет у: {args.data_dir}/train та {args.data_dir}/val")


if __name__ == "__main__":
    main()
