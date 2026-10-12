
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CNN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])


def load_models():
    """Tải 5 mô hình đã huấn luyện."""
    cnn = SimpleCNN()

    state = torch.load(
        MODEL_DIR / "simple_cnn.pth",
        map_location="cpu",
        weights_only=True
    )
    cnn.load_state_dict(state)
    cnn.eval()

    models = {
        "SimpleCNN": cnn,
        "Naive Bayes": joblib.load(
            MODEL_DIR / "naive_bayes.joblib"
        ),
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.joblib"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),
    }

    return models


def predict_image(image, models):
    """Dự đoán cùng một ảnh bằng cả 5 mô hình."""

    image = image.convert("RGB").resize((32, 32))

    # CNN: giữ cấu trúc ảnh 3 x 32 x 32
    cnn_input = CNN_TRANSFORM(image).unsqueeze(0)

    # ML truyền thống: vector 3072 pixel, không normalize
    flat_input = (
        transforms.ToTensor()(image)
        .numpy()
        .flatten()
        .reshape(1, -1)
    )

    results = {}

    with torch.no_grad():
        logits = models["SimpleCNN"](cnn_input)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred = int(np.argmax(probs))

    results["SimpleCNN"] = {
        "label": CLASSES[pred],
        "confidence": float(probs[pred]),
        "probabilities": probs.tolist()
    }

    for name, model in models.items():
        if name == "SimpleCNN":
            continue

        probs = model.predict_proba(flat_input)[0]
        pred = int(np.argmax(probs))
        class_id = int(model.classes_[pred])

        results[name] = {
            "label": CLASSES[class_id],
            "confidence": float(probs[pred]),
            "probabilities": probs.tolist()
        }

    return results
