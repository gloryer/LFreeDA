import argparse
import os
import time
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils.utils import load_image_malware, load_image_normal, change_attack_label, MacroF1

project_root = Path(__file__).resolve().parent.parent.parent.parent

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()
    start_time = time.time()

    label_path = str(project_root / "data/labels/mb24/March/march_malware.csv")
    img_path = str(project_root / "data/image_features/mb24/march/march")
    malware_march_x, malware_march_y = load_image_malware(img_path, label_path)

    label_path = str(project_root / "data/labels/mb24/April/april_malware.csv")
    img_path = str(project_root / "data/image_features/mb24/april/april")
    malware_april_x, malware_april_y = load_image_malware(img_path, label_path)

    label_path = str(project_root / "data/labels/mb24/May/may_malware.csv")
    img_path = str(project_root / "data/image_features/mb24/may/may")
    malware_may_x, malware_may_y = load_image_malware(img_path, label_path)

    label_path = str(project_root / "data/labels/mb24/Nov/nov_malware.csv")
    img_path = str(project_root / "data/image_features/mb24/nov")
    malware_nov_x, malware_nov_y = load_image_malware(img_path, label_path)

    label_path = str(project_root / "data/labels/mb24/Dec/dec_malware.csv")
    img_path = str(project_root / "data/image_features/mb24/dec")
    malware_dec_x, malware_dec_y = load_image_malware(img_path, label_path)

    # Load benign_source datasets 1-4 (all-benign folders, label 0 for every image)
    benign_source_dataset1_x = load_image_normal(str(project_root / "data/image_features/benign_source/dataset1"))
    benign_source_dataset1_y = np.zeros((len(benign_source_dataset1_x), 1))

    benign_source_dataset2_x = load_image_normal(str(project_root / "data/image_features/benign_source/dataset2"))
    benign_source_dataset2_y = np.zeros((len(benign_source_dataset2_x), 1))

    benign_source_dataset3_x = load_image_normal(str(project_root / "data/image_features/benign_source/dataset3"))
    benign_source_dataset3_y = np.zeros((len(benign_source_dataset3_x), 1))

    benign_source_dataset4_x = load_image_normal(str(project_root / "data/image_features/benign_source/dataset4"))
    benign_source_dataset4_y = np.zeros((len(benign_source_dataset4_x), 1))

    # Merge benign_source datasets
    source_normal_x = np.concatenate([benign_source_dataset1_x, benign_source_dataset2_x,
                                       benign_source_dataset3_x, benign_source_dataset4_x], axis=0)
    source_normal_y = np.concatenate([benign_source_dataset1_y, benign_source_dataset2_y,
                                       benign_source_dataset3_y, benign_source_dataset4_y], axis=0)

    # Load benign_target datasets 1-4 (all-benign folders, label 0 for every image)
    benign_target_dataset1_x = load_image_normal(str(project_root / "data/image_features/benign_target/dataset1"))
    benign_target_dataset1_y = np.zeros((len(benign_target_dataset1_x), 1))

    benign_target_dataset2_x = load_image_normal(str(project_root / "data/image_features/benign_target/dataset2"))
    benign_target_dataset2_y = np.zeros((len(benign_target_dataset2_x), 1))

    benign_target_dataset3_x = load_image_normal(str(project_root / "data/image_features/benign_target/dataset3"))
    benign_target_dataset3_y = np.zeros((len(benign_target_dataset3_x), 1))

    benign_target_dataset4_x = load_image_normal(str(project_root / "data/image_features/benign_target/dataset4"))
    benign_target_dataset4_y = np.zeros((len(benign_target_dataset4_x), 1))

    # Merge benign_target datasets
    target_normal_x = np.concatenate([benign_target_dataset1_x, benign_target_dataset2_x,
                                       benign_target_dataset3_x, benign_target_dataset4_x], axis=0)
    target_normal_y = np.concatenate([benign_target_dataset1_y, benign_target_dataset2_y,
                                       benign_target_dataset3_y, benign_target_dataset4_y], axis=0)

    source_malware_x = np.concatenate((malware_march_x, malware_april_x, malware_may_x), axis=0)
    source_malware_y = np.concatenate((malware_march_y, malware_april_y, malware_may_y), axis=0)

    target_malware_train_x = malware_nov_x
    target_malware_train_y = malware_nov_y

    target_malware_test_x = malware_dec_x
    target_malware_test_y = malware_dec_y

    target_normal_train_x, target_normal_test_x, target_normal_train_y, target_normal_test_y = train_test_split(target_normal_x, target_normal_y, test_size=0.5, random_state=42)

    source_malware_y = np.apply_along_axis(change_attack_label, 1, source_malware_y)
    target_malware_train_y = np.apply_along_axis(change_attack_label, 1, target_malware_train_y)
    target_malware_test_y = np.apply_along_axis(change_attack_label, 1, target_malware_test_y)

    source_x = np.concatenate((source_malware_x, source_normal_x), axis=0)
    source_y = np.concatenate((source_malware_y, source_normal_y), axis=0)

    target_x_train = np.concatenate((target_malware_train_x, target_normal_train_x), axis=0)
    target_y_train = np.concatenate((target_malware_train_y, target_normal_train_y), axis=0)

    target_x_test = np.concatenate((target_malware_test_x, target_normal_test_x), axis=0)
    target_y_test = np.concatenate((target_malware_test_y, target_normal_test_y), axis=0)

    # One-hot encode labels
    source_y = tf.keras.utils.to_categorical(source_y, num_classes=2)
    target_y_train = tf.keras.utils.to_categorical(target_y_train, num_classes=2)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=2)

    source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(source_x, source_y, test_size=0.25, random_state=42)

    ################################################################################
    # Config
    ################################################################################
    learning_rate_1 = 1e-3
    num_classes = 2

    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

        x_train = np.concatenate((source_x_train, target_x_train), axis=0)
        y_train = np.concatenate((source_y_train, target_y_train), axis=0)

        model = Sequential([
            ResNet50(include_top=False, weights="imagenet"),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dense(2, activation="softmax")
        ])

        optimizer_1 = Adam(learning_rate_1)

        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc", MacroF1(num_classes)])

        model.fit(x_train, y_train, batch_size=32, epochs=args.epochs, validation_data=(target_x_test, target_y_test))

        # Evaluate the model on the test data
        print("Evaluate on test data")
        results = model.evaluate(target_x_test, target_y_test, batch_size=16)
        print("test loss, test acc, test f1:", results)

        elapsed = time.time() - start_time
        print("Total runtime: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
