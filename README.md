# 🦋 ButterflyVision: U-Net Based Butterfly Image Segmentation

## Project Overview

ButterflyVision is a deep learning project that performs semantic segmentation of butterfly images using the U-Net architecture.

The model predicts pixel-level masks that accurately identify butterfly regions from natural images.

---

## Objectives

- Detect butterfly regions in images.
- Generate accurate segmentation masks.
- Automate image analysis for biodiversity research.
- Explore deep learning based semantic segmentation.

---

## Dataset

The dataset contains:

- RGB Butterfly Images
- Ground Truth Masks

Dataset Structure:

```text
images/
├── 0010001.png
├── 0010002.png
└── ...

Masks/
├── 0010001_seg0.png
├── 0010002_seg0.png
└── ...
```

---

## Methodology

### Data Preprocessing

- Image resizing
- Normalization
- Mask preparation
- Train-validation split

### Model

U-Net Architecture

Components:

- Encoder
- Bottleneck
- Decoder
- Skip Connections

### Training

- Optimizer: Adam
- Loss Function: Binary Cross Entropy
- Evaluation Metrics:
  - Accuracy
  - IoU
  - Dice Score

### Post Processing

- Thresholding
- Noise Removal
- Mask Refinement

---

## Project Structure

```text
butterfly_segmentation/
│
├── data_preprocessing.py
├── model_training.py
├── post_processing.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── sample_results/
```

---

## Installation

```bash
git clone <repository-url>

cd ButterflyVision-U-Net-Segmentation

pip install -r requirements.txt
```

---

## Usage

### Train Model

```bash
python model_training.py
```

### Run Full Pipeline

```bash
python main.py
```

---

## Results
```text
butterfly_segmentation/
│
├── sample_results

```
---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Future Work

- Attention U-Net
- DeepLabV3+
- Transfer Learning
- Real-Time Segmentation

---

## Author

Himanshi Bhamu
