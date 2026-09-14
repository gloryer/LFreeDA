import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import sys
sys.path.append('StepI')

from utils import load_image_malware, load_image_normal, change_attack_label
from model import MaxDIrep

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

    print("Loading data ...")

    # Load malware data by month
    label_path = "../data/labels/mb24/March/march_malware.csv"
    img_path = "../data/image_features/mb24/march/march"
    malware_march_x, malware_march_y = load_image_malware(img_path, label_path)

    label_path = "../data/labels/mb24/April/april_malware.csv"
    img_path = "../data/image_features/mb24/april/april"
    malware_april_x, malware_april_y = load_image_malware(img_path, label_path)

    label_path = "../data/labels/mb24/May/may_malware.csv"
    img_path = "../data/image_features/mb24/may/may"
    malware_may_x, malware_may_y = load_image_malware(img_path, label_path)

    label_path = "../data/labels/mb24/July/july_malware.csv"
    img_path = "../data/image_features/mb24/july/july"
    malware_july_x, malware_july_y = load_image_malware(img_path, label_path)

    label_path = "../data/labels/mb24/Aug/aug_malware.csv"
    img_path = "../data/image_features/mb24/aug/aug"
    malware_aug_x, malware_aug_y = load_image_malware(img_path, label_path)

    # Load normal data source
    img_path = "../data/image_features/benign_source/dataset1"
    source_normal_x_1 = load_image_normal(img_path)
    source_normal_y_1 = np.zeros((source_normal_x_1.shape[0], 1))

    img_path = "../data/image_features/benign_source/dataset2"
    source_normal_x_2 = load_image_normal(img_path)
    source_normal_y_2 = np.zeros((source_normal_x_2.shape[0], 1))

    img_path = "../data/image_features/benign_source/dataset3"
    source_normal_x_3 = load_image_normal(img_path)
    source_normal_y_3 = np.zeros((source_normal_x_3.shape[0], 1))

    img_path = "../data/image_features/benign_source/dataset4"
    source_normal_x_4 = load_image_normal(img_path)
    source_normal_y_4 = np.zeros((source_normal_x_4.shape[0], 1))

    # Merge source normal data
    source_normal_x = np.concatenate((source_normal_x_1, source_normal_x_2, source_normal_x_3, source_normal_x_4), axis=0)
    source_normal_y = np.concatenate((source_normal_y_1, source_normal_y_2, source_normal_y_3, source_normal_y_4), axis=0)

    # Load normal data target
    img_path = "../data/image_features/benign_target/dataset1"
    target_normal_x_1 = load_image_normal(img_path)
    target_normal_y_1 = np.zeros((target_normal_x_1.shape[0], 1))

    img_path = "../data/image_features/benign_target/dataset2"
    target_normal_x_2 = load_image_normal(img_path)
    target_normal_y_2 = np.zeros((target_normal_x_2.shape[0], 1))

    img_path = "../data/image_features/benign_target/dataset3"
    target_normal_x_3 = load_image_normal(img_path)
    target_normal_y_3 = np.zeros((target_normal_x_3.shape[0], 1))

    img_path = "../data/image_features/benign_target/dataset4"
    target_normal_x_4 = load_image_normal(img_path)
    target_normal_y_4 = np.zeros((target_normal_x_4.shape[0], 1))

    # Merge target normal data
    target_normal_x = np.concatenate((target_normal_x_1, target_normal_x_2, target_normal_x_3, target_normal_x_4), axis=0)
    target_normal_y = np.concatenate((target_normal_y_1, target_normal_y_2, target_normal_y_3, target_normal_y_4), axis=0)

    print("Data loaded")

    # Source malware data
    source_malware_x = np.concatenate((malware_march_x, malware_april_x, malware_may_x), axis=0)
    source_malware_y = np.concatenate((malware_march_y, malware_april_y, malware_may_y), axis=0)

    # Target malware train data
    target_malware_train_x = malware_july_x
    target_malware_train_y = malware_july_y

    # Target malware test data
    target_malware_test_x = malware_aug_x
    target_malware_test_y = malware_aug_y

    # Target normal train and test data
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
    print("Target origin : {}".format(target_normal_x.shape))
    print("Target origin : {}".format(target_normal_y.shape))
    print("Source {}".format(source_normal_x.shape))
    print("Source {}".format(source_normal_y.shape))

    print("=============================================")

    print("Normal data target train...")
    print("Target origin : {}".format(target_normal_train_x.shape))
    print("Target origin : {}".format(target_normal_train_y.shape))

    print("=============================================")


    print("Normal data target test...")
    print("Target origin : {}".format(target_normal_test_x.shape))
    print("Target origin : {}".format(target_normal_test_y.shape))

    print("=============================================")

    # change labels to 1 for malware
    source_malware_y = np.apply_along_axis(change_attack_label, 1, source_malware_y)
    target_malware_train_y = np.apply_along_axis(change_attack_label, 1, target_malware_train_y)
    target_malware_test_y = np.apply_along_axis(change_attack_label, 1, target_malware_test_y)

    # combine source malware and source normal data
    source_x = np.concatenate((source_malware_x, source_normal_x), axis=0)
    source_y = np.concatenate((source_malware_y, source_normal_y), axis=0)

    # combine target train malware and target train normal data
    target_x_train = np.concatenate((target_malware_train_x, target_normal_train_x), axis=0)
    target_y_train = np.concatenate((target_malware_train_y, target_normal_train_y), axis=0)

    # combine target test malware and target test normal data
    target_x_test = np.concatenate((target_malware_test_x, target_normal_test_x), axis=0)
    target_y_test = np.concatenate((target_malware_test_y, target_normal_test_y), axis=0)

    # one-hot encode labels
    source_y = tf.keras.utils.to_categorical(source_y, num_classes=2)
    target_y_train = tf.keras.utils.to_categorical(target_y_train, num_classes=2)
    target_y_test = tf.keras.utils.to_categorical(target_y_test, num_classes=2)

    # Split source data into train and test sets
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

    learning_rate_2 = 0.0001
    epochs = 30  # Number of training epochs
    n_class = 2
    input_shape = source_x.shape[1]

    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

        model = MaxDIrep(source_x_train, source_y_train, target_x_train, target_y_train,
                        source_x_test, source_y_test, target_x_test, target_y_test, epochs=30)

        generator, classifier = model.train()

        y_target_class_pred = classifier.predict(generator(target_x_test)).argmax(1)

        result = accuracy_score(target_y_test.argmax(1), y_target_class_pred)

        print("The test acc is {}".format(result))

        # Uncomment below to save the trained Step I model. Saved under results/
        # (not data/) so a from-scratch run never overwrites the precomputed
        # data/stepI_trained_models/ shipped with this repo.
        # save_dir = "../results/stepI_trained_models_scratch/mb24/aug"
        # os.makedirs(save_dir, exist_ok=True)
        # generator.save(os.path.join(save_dir, "generator"))
        # classifier.save(os.path.join(save_dir, "classifier"))
        # print("Saved Step I model to {}".format(save_dir))
