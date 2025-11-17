import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,  GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from Utils.utils import load_image, f1_m
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == "__main__":





    # Load source train data
    data = np.load('../../../data/stepII_constructed_datasets/big15/C1/source_train.npz', allow_pickle=True)
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


    # Load source test data
    data = np.load('../../../data/stepII_constructed_datasets/big15/C1/source_test.npz', allow_pickle=True)
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


    # Load selected target train data with pseudo-labels
    data = np.load('../../../data/stepII_constructed_datasets/big15/C1/target_train_filtered.npz', allow_pickle=True)
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


    # Load target test data
    data = np.load('../../../data/stepII_constructed_datasets/big15/C1/target_test.npz', allow_pickle=True)
    target_path_test = data['target_path_test']   # shape (N,), dtype object
    target_y_test = data['target_y_test']      # shape (N,)


    target_x_test, target_y_test  = load_image(target_path_test, target_y_test)

    uniq, cnts = np.unique(target_y_test, return_counts=True)
    num_classes = len(uniq)
    counts = np.zeros(num_classes, dtype=int)

    counts[uniq] = cnts

    for cls, cnt in enumerate(counts):
        print(f"Class {cls}: {cnt} samples")

    
    print("Loaded source_x_test shape:", target_x_test.shape)
    print("Loaded target_y_test shape:", target_y_test.shape)


    source_y_train = tf.keras.utils.to_categorical(source_y_train, num_classes = 2)
    source_y_test = tf.keras.utils.to_categorical(source_y_test, num_classes = 2)
    target_y_train_filtered = tf.keras.utils.to_categorical(target_y_train_filtered, num_classes = 2)
    target_y_test = tf.keras.utils.to_categorical(target_y_test , num_classes = 2)


    learning_rate_1 = 1e-3  # Learning rate



    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

    
        x_train = np.concatenate((source_x_train, target_x_train_filtered), axis = 0)
        y_train = np.concatenate((source_y_train, target_y_train_filtered), axis = 0)
        


        model = Sequential([
            ResNet50(include_top=False, weights="imagenet"),
            GlobalAveragePooling2D(),
            Dense(256, activation = "relu"),
            Dense(2, activation = "softmax")
        ])
        
        optimizer_1 = Adam(learning_rate_1)
        

        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc" ,f1_m])

        model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(target_x_test, target_y_test))


        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model.evaluate(target_x_test, target_y_test, batch_size=16)
        print("test loss, test acc, test f1:", results)

    
    
    