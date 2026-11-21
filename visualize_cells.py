
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

# Dataset location
dataset_path = Path(
    "data/raw/pbc_dataset/PBC_dataset_normal_DIB"
)

# Five WBC types
cell_types = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil"
]

# Create a figure
fig, axes = plt.subplots(1, 5, figsize=(16, 5))

for ax, cell in zip(axes, cell_types):

    folder = dataset_path / cell

    # Find JPG images
    images = list(folder.glob("*.jpg"))

    # Select a random image
    image_path = random.choice(images)

    # Open image
    image = Image.open(image_path)

    # Display image
    ax.imshow(image)
    ax.set_title(cell.capitalize())
    ax.axis("off")

plt.tight_layout()
plt.show()

