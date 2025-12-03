# Monkeypox 4-Class Classification Project

This project implements a modular deep learning pipeline for classifying 4 types of skin conditions: Monkeypox, Chickenpox, Measles, and Normal.

## Project Structure

```
project_root/
├── configs/                # Configuration files
├── src/
│   ├── data/               # Dataset building logic
│   ├── models/             # Model definitions
│   ├── training/           # Training loop logic
│   ├── inference/          # Inference logic
│   ├── evaluation/         # Evaluation logic
│   └── utils/              # Utilities (config, transforms, seed)
├── scripts/
│   └── split_data.py       # Data splitting script
├── train.py                # Main training script
├── evaluate.py             # Main evaluation script
├── predict.py              # Main prediction script
└── README.md
```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch torchvision pyyaml tqdm scikit-learn pillow
   ```

## Usage

### 1. Prepare Data

The dataset is expected to be at `./dataset/Monkeypox Skin Image Dataset/`.

Run the split script to organize the data into `data/train`, `data/val`, and `data/test`:

```bash
python scripts/split_data.py
```

### 2. Training

To train the model using the default configuration:

```bash
python train.py --config configs/default.yaml
```

This will save the best model to `checkpoints/best_model.pth`.

### 3. Evaluation

To evaluate the trained model on the test set:

```bash
python evaluate.py --config configs/default.yaml --model_path checkpoints/best_model.pth
```

### 4. Inference

To predict the class of a single image:

```bash
python predict.py path/to/image.jpg --model_path checkpoints/best_model.pth
```

## Google Colab

1. Upload the entire project folder to Google Drive.
2. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/path/to/project
   ```
3. Install dependencies.
4. Run `python scripts/split_data.py` (if data is not yet split).
5. Run `python train.py --config configs/colab.yaml`.
