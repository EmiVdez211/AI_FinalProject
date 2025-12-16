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

## Dataset Description

The dataset is organized into training and validation sets.  
Each class is represented by its own folder, and labels are inferred automatically from directory names.

data/
├── train/
│ ├── cola/
│ ├── orange_juice/
│ └── water/
└── val/
├── cola/
├── orange_juice/
└── water/


Images are stored in standard formats such as JPEG and PNG, which are fully supported by TensorFlow image pipelines.

---

## Model Architecture

The project uses MobileNetV2, a lightweight and efficient convolutional neural network pretrained on the ImageNet dataset.

Key characteristics of MobileNetV2:
- Pretrained on millions of images  
- Efficient and suitable for academic projects  
- Strong feature extraction capabilities  

The pretrained network is used as a frozen feature extractor, and a custom classification head is added for the 3-class beverage classification task.

---

## Transfer Learning Strategy

- The convolutional base (MobileNetV2) is frozen to preserve learned features  
- A custom classification head is trained, consisting of:
  - Global Average Pooling  
  - Dropout layer for regularization  
  - Dense softmax layer for multi-class classification  
- Only the newly added layers are trained  

This strategy ensures stable training and reduces overfitting.

---

## Project Structure

transfer-learning-beverage-classifier/
├── data/
├── models/
├── reports/
├── src/
│ ├── config.py
│ ├── data.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md


- `train.py` handles model training and saving  
- `predict.py` performs inference on new images  
- `model.py` defines the CNN architecture  
- `data.py` loads and prepares datasets  
- `config.py` centralizes configuration and paths using `Path(__file__).parent`  

---

## Installation

### Requirements
- Python 3.11 or higher  
- Virtual environment recommended  

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

### Training the Model

To train the model, ensure images are correctly placed in the dataset folders and run:

python -m src.train

This process:

Trains the CNN model

Saves the trained model in models/

Stores training metrics and reports

### Running Predictions

To classify a new image:

python -m src.predict --image path/to/image.jpg

The output includes:

Predicted class

Confidence score

Probability distribution across all classes

### Results

The trained model successfully classifies beverage images into the three defined categories.
Prediction confidence varies depending on image quality, lighting conditions, and background, reflecting realistic real-world performance.