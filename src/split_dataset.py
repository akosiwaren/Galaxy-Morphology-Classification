
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset path
file_path = "dataset/Galaxy10_DECals_NoDuplicated.h5"

# Load labels
with h5py.File(file_path, "r") as file:
    labels = np.array(file["ans"])

# Create image indices
indices = np.arange(len(labels))

# First split: 70% training, 30% temporary
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    indices,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Second split: 15% validation, 15% testing
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

# Display split information
print("GALAXY10 DATASET SPLITTING")
print("--------------------------")

print(f"Total images: {len(indices)}")
print(f"Training images: {len(train_indices)}")
print(f"Validation images: {len(val_indices)}")
print(f"Testing images: {len(test_indices)}")

# Check class distribution
class_names = [
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

print("\nTRAINING DISTRIBUTION")

for class_id, name in enumerate(class_names):
    count = np.sum(labels[train_indices] == class_id)
    print(f"{name}: {count}")

print("\nVALIDATION DISTRIBUTION")

for class_id, name in enumerate(class_names):
    count = np.sum(labels[val_indices] == class_id)
    print(f"{name}: {count}")

print("\nTESTING DISTRIBUTION")

for class_id, name in enumerate(class_names):
    count = np.sum(labels[test_indices] == class_id)
    print(f"{name}: {count}")

# Save split indices for future use
np.savez(
    "dataset/galaxy10_splits.npz",
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
)

print("\nDataset splitting completed!")
print("Split indices saved successfully.")
