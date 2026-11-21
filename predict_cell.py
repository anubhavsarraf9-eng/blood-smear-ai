import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

IMAGE_SIZE = 128

MODEL_PATH = Path(__file__).resolve().parent / "best_blood_cell_model.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


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


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

model = BloodCellCNN(len(classes)).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


def predict_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_cell = classes[predicted_index.item()]

    confidence_score = confidence.item() * 100

    all_scores = {
        cell: probabilities[0][i].item() * 100
        for i, cell in enumerate(classes)
    }

    return predicted_cell, confidence_score, all_scores
