"""
Script to check unique labels in segmentation tiffs.
Reads individual tiffs into tiff stack before finding the number of unique labels.
Note: Only use volumes that fit into your RAM.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University, UK
"""

import os
import numpy as np
from PIL import Image
import tifffile as t


def read_tiff_stack(folder_path):
    # Get a list of all TIFF files in the folder
    tiff_files = [
        f for f in os.listdir(folder_path) if f.endswith(".tiff") or f.endswith(".tif")
    ]
    tiff_files = sorted(tiff_files) # you can print this for sanity check, imp. when visualising!!

    # Read and stack the TIFF images
    image_stack = []
    for tiff_file in tiff_files:
        image_path = os.path.join(folder_path, tiff_file)
        image_array = t.imread(image_path)
        # image = Image.open(image_path)
        # image_array = np.array(image)
        image_stack.append(image_array)

    # Stack the images along a new axis to create a 3D array
    stack_array = np.stack(image_stack, axis=0)
    return stack_array


def count_unique_labels(image_stack):
    # Flatten the stack to count unique labels
    flattened_stack = image_stack.flatten()
    unique_labels = np.unique(flattened_stack)
    return len(unique_labels), unique_labels


# Specify the folder containing the TIFF images
folder_path = (
    "/path/to/folder" <-- FIXME
)

# Read the TIFF stack
tiff_stack = read_tiff_stack(folder_path)

# Count the unique labels in the stack
num_unique_labels, unique_labels = count_unique_labels(tiff_stack)

print(f"Number of unique labels: {num_unique_labels}")
print(f"Unique labels: {unique_labels}")