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

    # CREATE BAR CHART
    plt.figure(figsize=(13, 6))

    bars = plt.bar(
        [class_names[int(label)] for label in unique_labels],
        counts
    )

    plt.title("Galaxy10 Dataset Class Distribution")
    plt.xlabel("Galaxy Classes")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=45, ha="right")

    # Display count above each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()