# ANN Interactive Demo

An interactive demonstration of Artificial Neural Networks
for CIFAR-10 image classification.

## Features

- AI Battle Arena: Compare predictions from five models.
- Inside the CNN: Visualize convolutional feature maps.
- Model Performance: Compare classification accuracy and view a confusion matrix.
- Upload images or select images from CIFAR-10.

## Models

- SimpleCNN (PyTorch)
- Random Forest
- Logistic Regression
- Decision Tree
- Naive Bayes

## Dataset

CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

The dataset is not included in this repository.

## Installation

Requires Python 3.10.

```bash
pip install -r requirements.txt
```

Download the CIFAR-10 Python dataset and extract it into
`data/cifar-10-batches-py/`.

## Run

```bash
python -m streamlit run app.py
```

## Technologies

Python, PyTorch, scikit-learn, Streamlit, Plotly.

## Experimental Results

The recorded experiment used 5,000 training images,
1,000 test images, and 15 training epochs for SimpleCNN.

| Model | Accuracy |
|---|---:|
| SimpleCNN | 65.9% |
| Random Forest | 39.1% |
| Logistic Regression | 29.2% |
| Naive Bayes | 29.5% |
| Decision Tree | 22.5% |

These are results from the recorded experiment,
not necessarily from a new evaluation of the saved models.
