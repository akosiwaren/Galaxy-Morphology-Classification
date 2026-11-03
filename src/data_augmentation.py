
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================

FILE_PATH = "dataset/Galaxy10_DECals_NoDuplicated.h5"
SPLIT_PATH = "dataset/galaxy10_splits.npz"

CLASS_NAMES = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar-shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge"
]

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 16


# ==========================================
# 2. TRAINING AUGMENTATION
# ==========================================

train_transform = transforms.Compose([

    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270))
    ]),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 3. VALIDATION AND TEST TRANSFORM
# ==========================================

eval_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 4. CUSTOM GALAXY DATASET
# ==========================================

class GalaxyDataset(Dataset):

    def __init__(self, file_path, indices, transform=None):

        self.file_path = file_path
        self.indices = np.asarray(indices)
        self.transform = transform

        self.file = None

        # Read labels without keeping the HDF5 file open
        with h5py.File(self.file_path, "r") as h5_file:
            self.labels = np.asarray(h5_file["ans"])


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, idx):

        # Open HDF5 file lazily for this process
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        image_index = int(self.indices[idx])

        image = np.asarray(
            self.file["images"][image_index]
        )

        label = int(self.labels[image_index])

        # Convert image to PIL format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = Image.fromarray(image)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ==========================================
# 5. LOAD SAVED DATASET SPLITS
# ==========================================

splits = np.load(SPLIT_PATH)

train_indices = splits["train_indices"]
val_indices = splits["val_indices"]
test_indices = splits["test_indices"]


# ==========================================
# 6. CREATE DATASET OBJECTS
# ==========================================

train_dataset = GalaxyDataset(
    FILE_PATH,
    train_indices,
    transform=train_transform
)

val_dataset = GalaxyDataset(
    FILE_PATH,
    val_indices,
    transform=eval_transform
)

test_dataset = GalaxyDataset(
    FILE_PATH,
    test_indices,
    transform=eval_transform
)


# ==========================================
# 7. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 8. VERIFY DATASET
# ==========================================

if __name__ == "__main__":

    print("GALAXY10 DATA AUGMENTATION")
    print("--------------------------")

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    print(f"Batch size: {BATCH_SIZE}")

    images, labels = next(iter(train_loader))

    print("\nTRAINING BATCH CHECK")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image data type: {images.dtype}")
    print(f"Label data type: {labels.dtype}")

    print("\nSample labels:")

    for label in labels[:10]:
        print(f"{label.item()}: {CLASS_NAMES[label.item()]}")

    print("\nData augmentation setup completed!")


    # ==========================================
    # 9. VISUALIZE DATA AUGMENTATION
    # ==========================================

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):

        image, label = train_dataset[0]

        # Reverse normalization for visualization
        image = image * std + mean

        # Convert tensor to NumPy format
        image = image.permute(1, 2, 0).numpy()

        # Keep pixel values within 0–1
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.set_title(f"Variation {i + 1}")
        ax.axis("off")

    fig.suptitle(
        f"Data Augmentation: {CLASS_NAMES[label]}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()
