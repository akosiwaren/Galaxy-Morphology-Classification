import h5py

file_path = "dataset/Galaxy10_DECals_NoDuplicated.h5"

with h5py.File(file_path, "r") as file:

    print("Available datasets:", list(file.keys()))

    images = file["images"]
    labels = file["ans"]

    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print("Total images:", len(images))