
import random
import shutil
from pathlib import Path

# Make the results reproducible
random.seed(42)

# Original dataset
source = Path(
    "data/raw/pbc_dataset/PBC_dataset_normal_DIB"
)

# Destination folders
destination = Path("data/processed")

cell_types = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil"
]

# Split proportions
train_ratio = 0.70
val_ratio = 0.15

for cell in cell_types:

    print(f"\nProcessing {cell}...")

    folder = source / cell

    images = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():

        output_folder = destination / split_name / cell

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for image in split_images:
            shutil.copy2(
                image,
                output_folder / image.name
            )

        print(
            f"{split_name}: {len(split_images)} images"
        )

print("\nDataset preparation complete!")

