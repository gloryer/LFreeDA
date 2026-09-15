import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
sys.path.append('StepI')

from utils import load_image_malware, load_image_normal, change_attack_label

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Anchor data paths to this script's own location (not the current working
# directory), so this runs the same whether invoked as
# `python StepI/evaluate_pretrained_aug.py` from the repo root or
# `python evaluate_pretrained_aug.py` from inside StepI/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_MODEL_DIR = os.path.join(DATA_DIR, "stepI_trained_models", "mb24", "aug")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a (by default, the precomputed) Step I model on the July->Aug "
                    "task's target test set (Aug), reproducing Figure 3's Acc test target / "
                    "Macro F1 test target without training anything."
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Directory containing generator/ and classifier/ SavedModels "
             "(default: the precomputed aug model shipped with this repo, "
             "data/stepI_trained_models/mb24/aug/)"
    )
    args = parser.parse_args()

    print("Loading target test data (Aug malware + held-out benign_target split) ...")

    label_path = os.path.join(DATA_DIR, "labels", "mb24", "Aug", "aug_malware.csv")
    img_path = os.path.join(DATA_DIR, "image_features", "mb24", "aug", "aug")
    malware_aug_x, malware_aug_y = load_image_malware(img_path, label_path)

    target_normal_x_parts = []
    for i in range(1, 5):
        target_normal_x_parts.append(
            load_image_normal(os.path.join(DATA_DIR, "image_features", "benign_target", f"dataset{i}"))
        )
    target_normal_x = np.concatenate(target_normal_x_parts, axis=0)
    target_normal_y = np.zeros((target_normal_x.shape[0], 1))

    # Same 50/50 split (and seed) used to carve out the target test set during training
    # (see StepI/train_mb24+_aug.py) -- reproducing it here isolates the same test samples.
    _, target_normal_test_x, _, target_normal_test_y = train_test_split(
        target_normal_x, target_normal_y, test_size=0.5, random_state=42
    )

    target_malware_test_y = np.apply_along_axis(change_attack_label, 1, malware_aug_y)

    target_x_test = np.concatenate((malware_aug_x, target_normal_test_x), axis=0)
    target_y_test = np.concatenate((target_malware_test_y, target_normal_test_y), axis=0)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=2)

    print("Target test: {}".format(target_x_test.shape))

    print("Loading model from {} ...".format(args.model_dir))
    generator = tf.keras.models.load_model(os.path.join(args.model_dir, "generator"))
    classifier = tf.keras.models.load_model(os.path.join(args.model_dir, "classifier"))

    print("Evaluating ...")
    y_pred = classifier.predict(generator(target_x_test)).argmax(1)
    y_true = target_y_test.argmax(1)

    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100

    print("Acc test target: {:.2f}".format(acc))
    print("Macro F1 test target: {:.2f}".format(macro_f1))
    print("(Reference from Figure 3: Acc 80.8, Macro F1 77.7)")
