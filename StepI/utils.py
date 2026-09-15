import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

# Some malware-derived and benign images legitimately exceed Pillow's default
# decompression-bomb pixel-count threshold. Disable the check globally (once,
# for both loaders below) instead of per-call, and silence the corresponding
# warning so it doesn't spam stdout during normal loading.
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

def change_attack_label(x):
    label = [1.]
    return label

def _load_one_malware_image(args):
    filename, image_path = args
    hash_id = filename.split(".")[0]
    f = os.path.join(image_path, filename)
    image = Image.open(f).convert('RGB')
    image = image.resize((56, 56), Image.LANCZOS)
    image = np.array(image, dtype=int)
    return hash_id, image

def load_image_malware(image_path, label_path, max_workers=None):
    labels = pd.read_csv(label_path, header=0)
    # O(1) label lookup instead of re-scanning the whole dataframe per image
    label_map = dict(zip(labels["malware SHA-256"], labels["Label"]))

    filenames = [
        filename for filename in os.listdir(image_path)
        if filename.endswith(".png") and filename.split(".")[0] in label_map
    ]

    # Image decode/resize is I/O- and PIL-bound (PIL releases the GIL for
    # most of this work), so a thread pool parallelizes it well without the
    # process-pool overhead of pickling images back to the main process.
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda fn: _load_one_malware_image((fn, image_path)), filenames
        ))

    x = [image for _, image in results]
    y = [[label_map[hash_id]] for hash_id, _ in results]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x.astype('float32') / 255.

    return x, y

def _load_one_normal_image(args):
    filename, directory_path, image_size_limit = args
    file_path = os.path.join(directory_path, filename)
    try:
        with Image.open(file_path) as img:
            # Check if the image size is within the allowed limit
            if img.width * img.height <= image_size_limit:
                img = img.convert('RGB')
                img = img.resize((56, 56), Image.LANCZOS)
                return np.array(img, dtype=int)
            else:
                # Silently skip oversized images (still skipped, just not logged).
                pass
                return None
    except (Image.DecompressionBombError, OSError) as e:
        print(f"Error loading image {filename}: {e}")
        return None

def load_image_normal(directory_path, max_workers=None):
    image_list = []
    image_size_limit = 178956970  # Maximum allowed pixels per image

    filenames = [
        filename for filename in os.listdir(directory_path)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")
    ]

    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda fn: _load_one_normal_image((fn, directory_path, image_size_limit)), filenames
        ))

    image_list = [img for img in results if img is not None]

    image_list = np.asarray(image_list)
    image_list = image_list.astype('float32') / 255.

    return image_list
