# Transfer Learning Beverage Classifier using CNN

Universidad Tecnologica Metropolitana de Aguascalientes

Programming for Artificial Intelligence

Professor: Eng. Pablo Palacios Aranda

---

This project is a plug-and-play Python application that uses transfer learning with a pretrained Convolutional Neural Network (CNN) to classify three beverage categories:

- `cola`
- `orange_juice`
- `water`

The model is based on MobileNetV2 pretrained on ImageNet, and it is fine-tuned for this specific 3-class image classification task.

---

## How it works (short explanation)

A CNN learns visual patterns (edges, textures, shapes) from images. With transfer learning, we reuse a model that already learned strong generic visual features from a large dataset (ImageNet) and adapt it to our smaller, specific dataset.

In this project:
1. We load images from `data/train` and `data/val`.
2. We apply basic data augmentation.
3. We use MobileNetV2 as a frozen feature extractor.
4. We add a small classification head (pooling + dropout + dense softmax).
5. We train only the classification head to predict the beverage category.
 ---

## Dataset structure

data/
├─ train/
│ ├─ cola/
│ ├─ orange_juice/
│ └─ water/
└─ val/
├─ cola/
├─ orange_juice/
└─ water/

---

## Requirements

- Python 3.10+ recommended
- Dependencies listed in `requirements.txt`

Install dependencies:

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
