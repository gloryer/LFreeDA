import os
import numpy as np
import pandas as pd
from PIL import Image

def change_attack_label(x):
    label = [1.]
    return label

def load_image_malware(image_path, label_path):
    labels = pd.read_csv(label_path, header=0)

    x = []
    y = []

    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            hash_id = filename.split(".")[0]
            if hash_id in labels['malware SHA-256'].values:
                f = os.path.join(image_path, filename)
                image = Image.open(f).convert('RGB')
                image = image.resize((56, 56), Image.LANCZOS)
                image = np.array(image, dtype=int)
                x.append(image)
                y.append(labels[labels["malware SHA-256"] == hash_id]["Label"])

    x = np.asarray(x)
    y = np.asarray(y)

    x = x.astype('float32') / 255.

    return x, y

def load_image_normal(directory_path):
    image_list = []
    image_size_limit = 178956970  # Maximum allowed pixels per image

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory_path, filename)
            try:
                Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                with Image.open(file_path) as img:
                    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                    # Check if the image size is within the allowed limit
                    if img.width * img.height <= image_size_limit:
                        img = img.convert('RGB')
                        img = img.resize((56, 56), Image.LANCZOS)
                        img_array = np.array(img, dtype=int)
                        image_list.append(img_array)
                    else:
                        print(f"Image {filename} exceeds the size limit of {image_size_limit} pixels and will be skipped.")
            except (Image.DecompressionBombError, OSError) as e:
                print(f"Error loading image {filename}: {e}")

    image_list = np.asarray(image_list)
    image_list = image_list.astype('float32') / 255.

    return image_list
