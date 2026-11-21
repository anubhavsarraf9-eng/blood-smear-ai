
import os
from pathlib import Path

# Location of the original dataset
dataset_path = Path(
    "data/raw/pbc_dataset/PBC_dataset_normal_DIB"
)

# Five WBC categories
cell_types = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil"
]

print("\nPERIPHERAL BLOOD SMEAR DATASET")
print("=" * 40)

total_images = 0

for cell in cell_types:

    folder = dataset_path / cell

    if folder.exists():

        images = [
            f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        count = len(images)
        total_images += count

        print(f"{cell.capitalize():15} : {count}")

    else:
        print(f"{cell.capitalize():15} : Folder missing!")

print("=" * 40)
print("Total WBC images:", total_images)

