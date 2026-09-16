import argparse
import os
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utils.utils import load_image, MacroF1

# Get the project root directory relative to this script
project_root = Path(__file__).resolve().parent.parent.parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()

    # Load source train data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/sep/source_train.npz', allow_pickle=True)
    source_path_train = data['source_path_train']
    source_y_train = data['source_y_train']
    source_x_train, source_y_train = load_image(source_path_train, source_y_train)

    # Load source test data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/sep/source_test.npz', allow_pickle=True)
    source_path_test = data['source_path_test']
    source_y_test = data['source_y_test']
    source_x_test, source_y_test = load_image(source_path_test, source_y_test)

    # Load selected target train data with pseudo-labels
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/sep/target_train_filtered.npz', allow_pickle=True)
    target_path_train_filtered = data['target_path_train_filtered']
    target_pred_train_filtered = data['target_pred_train_filtered']
    target_x_train_filtered, target_y_train_filtered = load_image(target_path_train_filtered, target_pred_train_filtered)

    # Load target test data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/sep/target_test.npz', allow_pickle=True)
    target_path_test = data['target_path_test']
    target_y_test = data['target_y_test']
    target_x_test, target_y_test = load_image(target_path_test, target_y_test)

    num_classes = len(np.unique(target_y_test))

    # Convert labels to categorical
    source_y_train = tf.keras.utils.to_categorical(source_y_train, num_classes=num_classes)
    source_y_test = tf.keras.utils.to_categorical(source_y_test, num_classes=num_classes)
    target_y_train_filtered = tf.keras.utils.to_categorical(target_y_train_filtered, num_classes=num_classes)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=num_classes)

    # Config
    learning_rate_1 = 1e-3
    num_classes = 2

    for i in range(1):
        print(f"--------------------------{i} run-------------------------")

        # Combine source and target data
        x_train = np.concatenate((source_x_train, target_x_train_filtered), axis=0)
        y_train = np.concatenate((source_y_train, target_y_train_filtered), axis=0)

        # Build model
        model = Sequential([
            ResNet50(include_top=False, weights="imagenet"),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dense(2, activation="softmax")
        ])

        optimizer_1 = Adam(learning_rate_1)
        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc", MacroF1(num_classes)])
        model.fit(x_train, y_train, batch_size=32, epochs=args.epochs, validation_data=(target_x_test, target_y_test))

        # Evaluate model
        results = model.evaluate(target_x_test, target_y_test, batch_size=32)
        print("test loss, test acc, test macro_f1:", results)

        # sklearn sanity check
        try:
            y_prob = model.predict(target_x_test, batch_size=64)
            y_pred = np.argmax(y_prob, axis=1)
            y_true = np.argmax(target_y_test, axis=1)
            acc = accuracy_score(y_true, y_pred)
            macro = f1_score(y_true, y_pred, average="macro")
            micro = f1_score(y_true, y_pred, average="micro")
            weighted = f1_score(y_true, y_pred, average="weighted")
            cm = confusion_matrix(y_true, y_pred)
        except Exception as e:
            print("sklearn check skipped:", e)

