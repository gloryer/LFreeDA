import argparse
import os
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
from Utils.utils import load_image_malware, change_attack_label, MacroF1

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()

    label_path = "../../../data/labels/mb24/March/march_malware.csv"
    img_path = "../../../data/image_features/mb24/march/march"
    malware_march_x, malware_march_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/April/april_malware.csv"
    img_path = "../../../data/image_features/mb24/april/april"
    malware_april_x, malware_april_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/May/may_malware.csv"
    img_path = "../../../data/image_features/mb24/may/may"
    malware_may_x, malware_may_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/July/july_malware.csv"
    img_path = "../../../data/image_features/mb24/july/july"
    malware_july_x, malware_july_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Aug/aug_malware.csv"
    img_path = "../../../data/image_features/mb24/aug/aug"
    malware_aug_x, malware_aug_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/sep_malware.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    malware_sep_x, malware_sep_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Oct/oct_malware.csv"
    img_path = "../../../data/image_features/mb24/oct"
    malware_oct_x, malware_oct_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Nov/nov_malware.csv"
    img_path = "../../../data/image_features/mb24/nov"
    malware_nov_x, malware_nov_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Dec/dec_malware.csv"
    img_path = "../../../data/image_features/mb24/dec"
    malware_dec_x, malware_dec_y = load_image_malware(img_path, label_path)

    # Load benign_source datasets 1-4
    label_path = "../../../data/labels/mb24/Sep/benign_source_dataset1.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_source_dataset1_x, benign_source_dataset1_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_source_dataset2.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_source_dataset2_x, benign_source_dataset2_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_source_dataset3.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_source_dataset3_x, benign_source_dataset3_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_source_dataset4.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_source_dataset4_x, benign_source_dataset4_y = load_image_malware(img_path, label_path)

    # Merge benign_source datasets
    source_normal_x = np.concatenate([benign_source_dataset1_x, benign_source_dataset2_x,
                                       benign_source_dataset3_x, benign_source_dataset4_x], axis=0)
    source_normal_y = np.concatenate([benign_source_dataset1_y, benign_source_dataset2_y,
                                       benign_source_dataset3_y, benign_source_dataset4_y], axis=0)

    # Load benign_target datasets 1-4
    label_path = "../../../data/labels/mb24/Sep/benign_target_dataset1.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_target_dataset1_x, benign_target_dataset1_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_target_dataset2.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_target_dataset2_x, benign_target_dataset2_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_target_dataset3.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_target_dataset3_x, benign_target_dataset3_y = load_image_malware(img_path, label_path)

    label_path = "../../../data/labels/mb24/Sep/benign_target_dataset4.csv"
    img_path = "../../../data/image_features/mb24/sep/sep"
    benign_target_dataset4_x, benign_target_dataset4_y = load_image_malware(img_path, label_path)

    # Merge benign_target datasets
    target_normal_x = np.concatenate([benign_target_dataset1_x, benign_target_dataset2_x,
                                       benign_target_dataset3_x, benign_target_dataset4_x], axis=0)
    target_normal_y = np.concatenate([benign_target_dataset1_y, benign_target_dataset2_y,
                                       benign_target_dataset3_y, benign_target_dataset4_y], axis=0)

    source_malware_x = np.concatenate((malware_march_x, malware_april_x, malware_may_x), axis=0)
    source_malware_y = np.concatenate((malware_march_y, malware_april_y, malware_may_y), axis=0)

    target_malware_train_x = malware_aug_x
    target_malware_train_y = malware_aug_y

    target_malware_test_x = malware_sep_x
    target_malware_test_y = malware_sep_y

    target_normal_train_x, target_normal_test_x, target_normal_train_y, target_normal_test_y = train_test_split(target_normal_x, target_normal_y, test_size=0.5, random_state=42)

    print("Malware data ...")
    print("Target train: {}".format(target_malware_train_x.shape))
    print("Target train; {}".format(target_malware_train_y.shape))
    print("Target test: {}".format(target_malware_test_x.shape))
    print("Target test; {}".format(target_malware_test_y.shape))
    print("Source {}".format(source_malware_x.shape))
    print("Source {}".format(source_malware_y.shape))
    print("=============================================")

    print("Normal data ...")
    print("Target orgin : {}".format(target_normal_x.shape))
    print("Target orign : {}".format(target_normal_y.shape))
    print("Source {}".format(source_normal_x.shape))
    print("Source {}".format(source_normal_y.shape))

    print("=============================================")

    print("Normal data target train...")
    print("Target orgin : {}".format(target_normal_train_x.shape))
    print("Target orign : {}".format(target_normal_train_y.shape))

    print("=============================================")


    print("Normal data target test...")
    print("Target orgin : {}".format(target_normal_test_x.shape))
    print("Target orign : {}".format(target_normal_test_y.shape))

    print("=============================================")

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

    print("train test data ...")
    print("Target train: {}".format(target_x_train.shape))
    print("Target train: {}".format(target_y_train.shape))
    print("Target test: {}".format(target_x_test.shape))
    print("Target test: {}".format(target_y_test.shape))
    print("Source train: {}".format(source_x_train.shape))
    print("Source train: {}".format(source_y_train.shape))
    print("Source test: {}".format(source_x_test.shape))
    print("Source test: {}".format(source_y_test.shape))

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
