import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "dataset/Galaxy10_DECals_NoDuplicated.h5"

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

with h5py.File(file_path, "r") as file:

    labels = np.array(file["ans"])

    unique_labels, counts = np.unique(
        labels, return_counts=True
    )

    print("GALAXY10 CLASS DISTRIBUTION")

    for label, count in zip(unique_labels, counts):
        print(f"{class_names[int(label)]}: {count}")

    print("\nTotal Images:", len(labels))