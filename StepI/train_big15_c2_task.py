import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import sys
sys.path.append('StepI')

from utils import load_image_malware, load_image_normal, filter_label, change_attack_label
from model import MaxDIrep

os.environ["CUDA_VISIBLE_DEVICES"]="0"






if __name__ == "__main__":

    print("Loading data ...")

    # Load malware data
    label_path = "../data/labels/new_gmm_labels_id.csv"
    img_path ="../data/image_features/Big15_2"
    malware_x, malware_y =load_image_malware(img_path, label_path)
    
    # Load normal data source
    img_path ="../data/image_features/benign_source/dataset1"
    source_normal_x_1 =load_image_normal(img_path)
    source_normal_y_1 = np.zeros((source_normal_x_1.shape[0],1))

    img_path ="../data/image_features/benign_source/dataset2"
    source_normal_x_2 =load_image_normal(img_path)
    source_normal_y_2 = np.zeros((source_normal_x_2.shape[0],1))

    img_path ="../data/image_features/benign_source/dataset3"
    source_normal_x_3=load_image_normal(img_path)
    source_normal_y_3 = np.zeros((source_normal_x_3.shape[0],1))

    img_path ="../data/image_features/benign_source/dataset4"
    source_normal_x_4 =load_image_normal(img_path)
    source_normal_y_4 = np.zeros((source_normal_x_4.shape[0],1))

    #merge
    source_normal_x = np.concatenate((source_normal_x_1, source_normal_x_2, source_normal_x_3, source_normal_x_4), axis = 0)
    source_normal_y = np.concatenate((source_normal_y_1, source_normal_y_2, source_normal_y_3, source_normal_y_4), axis = 0)

    # Load normal data target
    img_path ="../data/image_features/benign_target/dataset1"
    target_normal_x_1 =load_image_normal(img_path)
    target_normal_y_1 = np.zeros((target_normal_x_1.shape[0],1))

    img_path ="../data/image_features/benign_target/dataset2"
    target_normal_x_2 =load_image_normal(img_path)
    target_normal_y_2 = np.zeros((target_normal_x_2.shape[0],1))

    img_path ="../data/image_features/benign_target/dataset3"
    target_normal_x_3=load_image_normal(img_path)
    target_normal_y_3 = np.zeros((target_normal_x_3.shape[0],1))

    img_path ="../data/image_features/benign_target/dataset4"
    target_normal_x_4 =load_image_normal(img_path)
    target_normal_y_4 = np.zeros((target_normal_x_4.shape[0],1))

    #merge
    target_normal_x = np.concatenate((target_normal_x_1, target_normal_x_2, target_normal_x_3, target_normal_x_4), axis = 0)
    target_normal_y = np.concatenate((target_normal_y_1, target_normal_y_2, target_normal_y_3, target_normal_y_4), axis = 0)
  

    print("Data loaded")


    # Filter source and target malware based on labels (C2)
    target_malware_x, target_malware_y, source_malware_x, source_malware_y =  filter_label(malware_x, malware_y, [2])
    print("Malware data ...")
    print("Target: {}".format(target_malware_x.shape))
    print("Target; {}".format(target_malware_y.shape))
    print("Source {}".format(source_malware_x.shape))
    print("Source {}".format(source_malware_y.shape))

    print("Normal data ...")
    print("Target: {}".format(target_normal_x.shape))
    print("Target: {}".format(target_normal_y.shape))
    print("Source {}".format(source_normal_x.shape))
    print("Source {}".format(source_normal_y.shape))

    # Change labels to 1 for malware and 0 for normal
    source_malware_y = np.apply_along_axis(change_attack_label, 1, source_malware_y)
    target_malware_y = np.apply_along_axis(change_attack_label, 1, target_malware_y)



    # Concatenate source malware and normal data
    source_x = np.concatenate((source_malware_x, source_normal_x), axis = 0)
    source_y = np.concatenate((source_malware_y, source_normal_y), axis = 0)

    # Concatenate target malware and normal data
    target_x = np.concatenate((target_malware_x, target_normal_x), axis = 0)
    target_y = np.concatenate((target_malware_y, target_normal_y), axis = 0)


    #one-hot encode labels
    source_y = tf.keras.utils.to_categorical(source_y, num_classes = 2)
    target_y = tf.keras.utils.to_categorical(target_y, num_classes = 2)


    print("Combined data ...")
    print("Target: {}".format(target_x.shape))
    print("Target: {}".format(target_y.shape))
    print("Source {}".format(source_x.shape))
    print("Source {}".format(source_y.shape))



    # Split data into train and test sets
    source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(source_x, source_y, test_size=0.25, random_state=42)
    target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(target_x, target_y, test_size=0.5, random_state=42)
        


    print("Train test data ...")
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


    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))


        model =MaxDIrep(source_x_train, source_y_train, target_x_train, target_y_train,
        source_x_test, source_y_test, target_x_test, target_y_test,  epochs=30)

        generator, classifier = model.train()

        y_target_class_pred = classifier.predict(generator(target_x_test)).argmax(1)

        result = accuracy_score(target_y_test.argmax(1), y_target_class_pred)

        print("The test acc is {}".format(result))

     







           


           
