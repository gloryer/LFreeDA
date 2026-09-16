import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Some malware-derived and benign images legitimately exceed Pillow's default
# decompression-bomb pixel-count threshold. Disable the check globally instead
# of per-call, and silence the corresponding warning so it doesn't spam stdout.
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# This file lives at <repo_root>/StepIII/Images/Utils/utils.py, so this is the
# repo root regardless of the caller's working directory. Step II saves image
# paths relative to the repo root (for portability across machines); resolve
# those against this when loading.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent



def encode(label_y, classes):
    
    one_hot = np.zeros((len(label_y), classes))
    

    for idx, label in enumerate(label_y):
        # Convert 1-based label to 0-based index
        one_hot[idx, label] = 1

    return one_hot  




def filter_label(df, y, label):
    #divide dataset based on the label[]
     
    output_df = df[np.where(y == label)[0]]
    
    res_df =  df[np.where(y != label)[0]]
    
    output_y = y[np.where(y == label)[0]]
    res_y = y[np.where(y != label)[0]]
    
    return  output_df, output_y, res_df, res_y

 


def obtain_labels(df, label_path):
    
    labels = pd.read_csv(label_path, header=0)
    
    y = []
    x = []
    
    for index, row in df.iterrows():
        #print(row["asm_id"].split(".")[0])
        hash_id = row["asm_id"].split(".")[0]
        if hash_id in labels['asm_id'].values: 
            row = row.drop("asm_id")
            x.append(row)
            y.append(labels[labels["asm_id"] == hash_id]["Class"])
            

    
    return np.array(x), np.array(y)


def change_attack_label(x):
    label = [1.]
    return label



def _load_one_image(args):
    path, label = args
    if not path.endswith(".png"):
        return None
    # Older constructed datasets stored absolute paths from whichever machine
    # built them; newer ones store paths relative to the repo root.
    full_path = path if os.path.isabs(path) else str(REPO_ROOT / path)
    image = Image.open(full_path).convert('RGB')
    image = image.resize((56, 56), Image.LANCZOS)
    image = np.array(image, dtype=int)
    return image, label


def load_image(image_path, labels, max_workers=None):
    # Image decode/resize is I/O- and PIL-bound (PIL releases the GIL for
    # most of this work), so a thread pool parallelizes it well without the
    # process-pool overhead of pickling images back to the main process.
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_load_one_image, zip(image_path, labels)))

    results = [r for r in results if r is not None]
    x = [image for image, _ in results]
    y = [label for _, label in results]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x.astype('float32') / 255.

    return x, y




# drop-in metric (same as I shared before)
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(
            name="cm", shape=(self.num_classes, self.num_classes),
            initializer="zeros", dtype=tf.float32
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        cm_batch = tf.math.confusion_matrix(
            y_true_labels, y_pred_labels,
            num_classes=self.num_classes, dtype=tf.float32
        )
        self.cm.assign_add(cm_batch)
    def result(self):
        TP = tf.linalg.tensor_diag_part(self.cm)
        FP = tf.reduce_sum(self.cm, axis=0) - TP
        FN = tf.reduce_sum(self.cm, axis=1) - TP
        precision = tf.math.divide_no_nan(TP, TP + FP)
        recall    = tf.math.divide_no_nan(TP, TP + FN)
        f1        = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)
    def reset_states(self):
        self.cm.assign(tf.zeros_like(self.cm))


def _load_one_malware_image(args):
    filename, image_path, label_by_hash = args
    if not filename.endswith(".png"):
        return None
    hash_id = filename.split(".")[0]
    if hash_id not in label_by_hash:
        return None
    f = os.path.join(image_path, filename)
    image = Image.open(f).convert('RGB')
    image = image.resize((56, 56), Image.LANCZOS)
    image = np.array(image, dtype=int)
    return image, label_by_hash[hash_id]


def load_image_malware(image_path, label_path, max_workers=None):
    labels = pd.read_csv(label_path, header=0)
    label_by_hash = dict(zip(labels["malware SHA-256"], labels["Label"]))

    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            _load_one_malware_image,
            ((filename, image_path, label_by_hash) for filename in os.listdir(image_path))
        ))

    results = [r for r in results if r is not None]
    x = [image for image, _ in results]
    y = [[label] for _, label in results]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x.astype('float32') / 255.

    return x, y




def _load_one_normal_image(args):
    filename, directory_path, image_size_limit = args
    if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
        return None
    file_path = os.path.join(directory_path, filename)
    try:
        with Image.open(file_path) as img:
            if img.width * img.height > image_size_limit:
                return None
            img = img.convert('RGB')
            img = img.resize((56, 56), Image.LANCZOS)
            return np.array(img, dtype=int)
    except (Image.DecompressionBombError, OSError) as e:
        print(f"Error loading image {filename}: {e}")
        return None


def load_image_normal(directory_path, max_workers=None):
    image_size_limit = 178956970  # Maximum allowed pixels per image

    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            _load_one_normal_image,
            ((filename, directory_path, image_size_limit) for filename in os.listdir(directory_path))
        ))

    image_list = [r for r in results if r is not None]
    image_list = np.asarray(image_list)
    image_list = image_list.astype('float32') / 255.

    return image_list





    
