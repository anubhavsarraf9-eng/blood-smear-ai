
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# --------------------------------
# 1. CONFIGURATION
# --------------------------------

DATA_DIR = Path("data/processed")

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Select Apple GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --------------------------------
# 2. IMAGE PREPROCESSING
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# --------------------------------
# 3. LOAD DATASET
# --------------------------------

train_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform
)

val_data = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = train_data.classes
num_classes = len(classes)

print("Cell categories:", classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# --------------------------------
# 4. BUILD CNN MODEL
# --------------------------------

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
        x = self.classifier(x)

        return x


model = BloodCellCNN(num_classes).to(device)

# --------------------------------
# 5. HANDLE CLASS IMBALANCE
# --------------------------------

counts = torch.bincount(
    torch.tensor(train_data.targets)
).float()

class_weights = (
    len(train_data) / (num_classes * counts)
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------
# 6. TRAINING LOOP
# --------------------------------

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = 100 * correct / total

    # --------------------------------
    # 7. VALIDATION
    # --------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(
        f"Training Loss: "
        f"{running_loss / len(train_loader):.4f}"
    )

    print(
        f"Training Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss / len(val_loader):.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE
        }, "best_blood_cell_model.pth")

        print("Best model saved!")

print("\nTraining completed!")
print("Best validation accuracy:", best_accuracy)

