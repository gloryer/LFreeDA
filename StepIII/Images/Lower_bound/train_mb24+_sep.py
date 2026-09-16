import argparse
import os
import time
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config

# Add Utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils.utils import load_image, MacroF1

# Configure TensorFlow and GPU
np_config.enable_numpy_behavior()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()
    start_time = time.time()

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent.parent

    print("Loading data ...")

    # Load source data
    source_data = np.load(
        str(project_root / 'data/stepII_constructed_datasets/mb24/sep/source_train.npz'),
        allow_pickle=True
    )
    source_path_train = source_data['source_path_train']
    source_y_train = source_data['source_y_train']
    source_x_train, source_y_train = load_image(source_path_train, source_y_train)

    # Load target data
    target_data = np.load(
        str(project_root / 'data/stepII_constructed_datasets/mb24/sep/target_test.npz'),
        allow_pickle=True
    )
    target_path_test = target_data['target_path_test']
    target_y_test = target_data['target_y_test']
    target_x_test, target_y_test = load_image(target_path_test, target_y_test)

    # Convert labels to categorical
    source_y_train = tf.keras.utils.to_categorical(source_y_train, num_classes=2)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=2)

    # Configuration
    learning_rate = 1e-3
    num_classes = 2
    batch_size = 32
    epochs = args.epochs

    # Build and train model
    model = Sequential([
        ResNet50(include_top=False, weights='imagenet'),
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    optimizer = Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["acc", MacroF1(num_classes)]
    )

    print("Training model...")
    model.fit(
        source_x_train, source_y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(target_x_test, target_y_test)
    )

    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(target_x_test, target_y_test, batch_size=16)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]:.4f}")
    print(f"Test macro F1: {results[2]:.4f}")

    elapsed = time.time() - start_time
    print("Total runtime: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
