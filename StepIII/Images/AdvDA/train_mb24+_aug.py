import os
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from Utils.utils import load_image
from AdvDA.model import AdvDA_CNN
from sklearn.metrics import accuracy_score

# Get the project root directory relative to this script
project_root = Path(__file__).resolve().parent.parent.parent.parent

if __name__ == "__main__":

    #  Load source train data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/aug/source_train.npz', allow_pickle=True)
    source_path_train = data['source_path_train']   # shape (N,), dtype object
    source_y_train    = data['source_y_train']      # shape (N,)

    source_x_train, source_y_train = load_image(source_path_train, source_y_train  )


    uniq, cnts = np.unique(source_y_train, return_counts=True)
    num_classes = len(uniq)
    counts = np.zeros(num_classes, dtype=int)

    counts[uniq] = cnts

    for cls, cnt in enumerate(counts):
        print(f"Class {cls}: {cnt} samples")

    print("Loaded source_x shape:", source_x_train.shape)
    print("Loaded source_y shape:", source_y_train.shape)


    #  Load source test data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/aug/source_test.npz', allow_pickle=True)
    source_path_test = data['source_path_test']   # shape (N,), dtype object
    source_y_test    = data['source_y_test']      # shape (N,)

    source_x_test, source_y_test = load_image(source_path_test, source_y_test)


    uniq, cnts = np.unique(source_y_test, return_counts=True)
    num_classes = len(uniq)
    counts = np.zeros(num_classes, dtype=int)

    counts[uniq] = cnts

   
    for cls, cnt in enumerate(counts):
        print(f"Class {cls}: {cnt} samples")


    print("Loaded source_x shape:", source_x_test.shape)
    print("Loaded source_y shape:", source_y_test.shape)


    #  Load selected target train data with pseudo-labels
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/aug/target_train_filtered.npz', allow_pickle=True)
    target_path_train_filtered = data['target_path_train_filtered']   # shape (N,), dtype object
    target_pred_train_filtered    = data['target_pred_train_filtered']      # shape (N,)


    target_x_train_filtered, target_y_train_filtered  = load_image(target_path_train_filtered,target_pred_train_filtered)


    uniq, cnts = np.unique(target_pred_train_filtered, return_counts=True)
    num_classes = len(uniq)
    counts = np.zeros(num_classes, dtype=int)

    counts[uniq] = cnts

    for cls, cnt in enumerate(counts):
        print(f"Class {cls}: {cnt} samples")

    
    print("Loaded target_x_train_filtered shape:", target_x_train_filtered.shape)
    print("Loaded target_y_train_filtered shape:", target_y_train_filtered.shape)


    #  Load target test data
    data = np.load(project_root / 'data/stepII_constructed_datasets/mb24/aug/target_test.npz', allow_pickle=True)
    target_path_test = data['target_path_test']   # shape (N,), dtype object
    target_y_test = data['target_y_test']      # shape (N,)


    target_x_test, target_y_test  = load_image(target_path_test, target_y_test)

    uniq, cnts = np.unique(target_y_test, return_counts=True)
    num_classes = len(uniq)
    counts = np.zeros(num_classes, dtype=int)

    counts[uniq] = cnts

    for cls, cnt in enumerate(counts):
        print(f"Class {cls}: {cnt} samples")

    
    print("Loaded target_x_test shape:", target_x_test.shape)
    print("Loaded target_y_test shape:", target_y_test.shape)


    source_y_train = tf.keras.utils.to_categorical(source_y_train, num_classes=num_classes)
    source_y_test = tf.keras.utils.to_categorical(source_y_test, num_classes=num_classes)
    target_y_train_filtered = tf.keras.utils.to_categorical(target_y_train_filtered, num_classes=num_classes)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=num_classes)


    #learning_rate_1 = 1e-3  # Learning rate
    learning_rate_2 = 0.0001
    epochs = 30  # Number of training epochs
    n_class = 2 






    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

        model = AdvDA_CNN(source_x_train, source_y_train, target_x_train_filtered, target_y_train_filtered,
                          source_x_test, source_y_test, target_x_test, target_y_test, epochs=30)

        generator, classifier = model.train()

        y_target_class_pred = classifier.predict(generator(target_x_test)).argmax(1)

        result = accuracy_score(target_y_test.argmax(1), y_target_class_pred)

        print("The test acc is {}".format(result))

    