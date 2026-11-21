
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

import matplotlib.pyplot as plt

# 1. Configuration

IMAGE_SIZE = 128
BATCH_SIZE = 16

DATA_DIR = Path("data/processed")
MODEL_PATH = "best_blood_cell_model.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Testing device:", device)

# 2. Image preprocessing

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 3. Load test dataset

def is_valid_image(path):
    filename = Path(path).name.lower()

    valid_extensions = (".jpg", ".jpeg", ".png")

    return (
        not filename.startswith(".")
        and filename.endswith(valid_extensions)
    )


test_data = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
    is_valid_file=is_valid_image
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 4. Define the same CNN architecture

class BloodCellCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# 5. Load the best saved model

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

assert test_data.classes == classes, (
    "Class order mismatch!"
)

model = BloodCellCNN(len(classes)).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")
print("Cell types:", classes)
print("Test images:", len(test_data))

# 6. Make predictions

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# 7. Calculate performance

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nTEST ACCURACY:")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

# 8. Confusion matrix

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Blood Cell Classification - Test Results")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved!")
print("Testing completed!")

