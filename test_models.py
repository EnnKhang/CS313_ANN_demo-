
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10

from src.inference import load_models, predict_image

print("Đang tải 5 mô hình...")
models = load_models()
print("Đã tải:", list(models.keys()))

data_dir = Path(__file__).resolve().parent / "data"

dataset = CIFAR10(
    root=str(data_dir),
    train=False,
    download=False
)

image, true_label = dataset[0]

print("\nNhãn thật:", dataset.classes[true_label])

results = predict_image(image, models)

for name, result in results.items():
    print(
        f"{name:<22} -> "
        f"{result['label']:<12} "
        f"Confidence: {result['confidence']:.2%}"
    )

print("\nTEST HOÀN THÀNH!")
