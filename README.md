# Класифікація зображень на базі CNN (ResNet18)

Проєкт для виробничої практики: система класифікації зображень на основі
згорткової нейронної мережі (transfer learning на ResNet18, попередньо
натренованому на ImageNet).

## Структура проєкту

```
cnn_project/
├── data/                    # датасет (train/ та val/, по папці на клас) — НЕ в git
├── src/
│   ├── prepare_dataset.py   # розбиття сирих зображень на train/val
│   ├── train.py             # тренування моделі
│   ├── evaluate.py          # оцінка: Accuracy, F1, Confusion Matrix
│   └── inference.py         # класифікація одного нового зображення
├── notebooks/
│   └── colab_train.ipynb    # готовий ноутбук для тренування на Google Colab (GPU)
├── outputs/                 # результати: модель, графіки, звіти (генеруються)
├── requirements.txt
├── .gitignore
└── README.md
```

## Швидкий старт (локально або в Colab)

### 1. Встановлення середовища
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Підготовка датасету
Розклади зображення так, щоб у кожній папці класу були лише фото цього класу:
```
raw/
    class1/  img1.jpg img2.jpg ...
    class2/  img1.jpg img2.jpg ...
```
Потім:
```bash
python src/prepare_dataset.py --raw_dir raw --data_dir data --val_split 0.2
```
Це автоматично створить `data/train/...` і `data/val/...` (80/20).

### 3. Тренування
```bash
python src/train.py --data_dir data --epochs 15 --batch_size 32
```
Результати з'являться в `outputs/`:
- `best_model.pth` — ваги найкращої моделі
- `training_curves.png` — графіки Loss/Accuracy по епохах
- `history.json` — сирі дані для звіту

### 4. Оцінка моделі
```bash
python src/evaluate.py --data_dir data --model_path outputs/best_model.pth
```
Створює `outputs/confusion_matrix.png` та `outputs/eval_report.txt`
(Accuracy, F1-score, precision/recall по кожному класу).

### 5. Інференс на новому зображенні
```bash
python src/inference.py --image шлях/до/фото.jpg --model_path outputs/best_model.pth
```

## Чому саме ResNet18 + transfer learning

Тренування CNN "з нуля" на невеликому датасеті (сотні-тисячі зображень) майже
завжди дає гірший результат і швидко перенавчається. ResNet18, попередньо
натренований на ImageNet (1.2 млн зображень, 1000 класів), уже "вміє" виділяти
універсальні візуальні ознаки (краї, текстури, форми). Ми заморожуємо його
згорткові шари (`freeze_backbone=True` в `train.py`) і донавчаємо лише
останній повнозв'язний шар під свої класи — це швидко і стабільно навіть
на невеликих датасетах.

Для звіту варто також описати альтернативу — тренування "з нуля" простої CNN
(2-3 згорткові шари) і порівняти результати з ResNet18 (це можна навести
як частину аналізу в розділі 1 методичного завдання).

## Для GPU без власного — Google Colab

У `notebooks/colab_train.ipynb` — готовий ноутбук: клонує репозиторій,
ставить залежності, тренує модель на безкоштовному GPU, зберігає результати
і пушить їх назад у GitHub.
