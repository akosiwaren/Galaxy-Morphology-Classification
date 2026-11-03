
import h5py
import numpy as np
import torch

# ==========================================
# 1. CONFIGURATION
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


# ==========================================
# 2. LOAD TRAINING LABELS
# ==========================================

with h5py.File(FILE_PATH, "r") as file:
    labels = np.asarray(file["ans"])

splits = np.load(SPLIT_PATH)

train_indices = splits["train_indices"]

train_labels = labels[train_indices]


# ==========================================
# 3. COUNT IMAGES PER CLASS
# ==========================================

class_counts = np.bincount(
    train_labels.astype(int),
    minlength=NUM_CLASSES
)

total_images = len(train_labels)


# ==========================================
# 4. CALCULATE CLASS WEIGHTS
# ==========================================

class_weights = total_images / (
    NUM_CLASSES * class_counts
)


# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("GALAXY10 CLASS WEIGHTS")
print("----------------------")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name}: "
        f"Images = {class_counts[i]}, "
        f"Weight = {class_weights[i]:.4f}"
    )

print("\nTotal training images:", total_images)


# ==========================================
# 6. CONVERT TO PYTORCH TENSOR
# ==========================================

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nPYTORCH CLASS WEIGHTS")
print(class_weights_tensor)


# ==========================================
# 7. SAVE CLASS WEIGHTS
# ==========================================

np.save(
    "dataset/galaxy10_class_weights.npy",
    class_weights
)

print("\nClass weights saved successfully!")
