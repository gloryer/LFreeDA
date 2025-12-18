import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from spektral.layers import DisjointLoader
np_config.enable_numpy_behavior()
import sys

# Add StepI to path for utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'StepI'))

from StepIII.CFGs.GraphMatching.graph_matching import GraphData
from StepIII.CFGs.Utils.utils import merge_dataset, binary_label, MacroF1
from model import GIN0

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

    print("Loading data ...")

    # Load malware data by month
    data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"

    # March
    path_0 = str(data_dir / "graph_features/mb24/March/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/March/1/cfg_embeddings")
    malware_march_0 = GraphData(path_0)
    malware_march_1 = GraphData(path_1)
    malware_march = merge_dataset(malware_march_0, malware_march_1)

    # April
    path_0 = str(data_dir / "graph_features/mb24/April/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/April/1/cfg_embeddings")
    malware_april_0 = GraphData(path_0)
    malware_april_1 = GraphData(path_1)
    malware_april = merge_dataset(malware_april_0, malware_april_1)

    # May
    path_0 = str(data_dir / "graph_features/mb24/May/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/May/1/cfg_embeddings")
    malware_may_0 = GraphData(path_0)
    malware_may_1 = GraphData(path_1)
    malware_may = merge_dataset(malware_may_0, malware_may_1)

    # July
    path_0 = str(data_dir / "graph_features/mb24/July/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/July/1/cfg_embeddings")
    malware_july_0 = GraphData(path_0)
    malware_july_1 = GraphData(path_1)
    malware_july = merge_dataset(malware_july_0, malware_july_1)

    # Aug
    path_0 = str(data_dir / "graph_features/mb24/Aug/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/Aug/1/cfg_embeddings")
    malware_aug_0 = GraphData(path_0)
    malware_aug_1 = GraphData(path_1)
    malware_aug = merge_dataset(malware_aug_0, malware_aug_1)

    # Sep
    path_0 = str(data_dir / "graph_features/mb24/Sep/0/cfg_embeddings")
    path_1 = str(data_dir / "graph_features/mb24/Sep/1/cfg_embeddings")
    malware_sep_0 = GraphData(path_0)
    malware_sep_1 = GraphData(path_1)
    malware_sep = merge_dataset(malware_sep_0, malware_sep_1)

    # Oct (single cfg_embeddings folder)
    path_oct = str(data_dir / "graph_features/mb24/Oct/cfg_embeddings")
    malware_oct = GraphData(path_oct)

    # Nov (single cfg_embeddings folder)
    path_nov = str(data_dir / "graph_features/mb24/Nov/cfg_embeddings")
    malware_nov = GraphData(path_nov)

    # Dec (single cfg_embeddings folder)
    path_dec = str(data_dir / "graph_features/mb24/Dec/cfg_embeddings")
    malware_dec = GraphData(path_dec)

    # Benign Source datasets
    benign_source_dataset1 = GraphData(str(data_dir / "graph_features/benign_source/dataset1/cfg_embeddings"))
    benign_source_dataset2 = GraphData(str(data_dir / "graph_features/benign_source/dataset2/cfg_embeddings"))
    benign_source_dataset3 = GraphData(str(data_dir / "graph_features/benign_source/dataset3/cfg_embeddings"))
    benign_source_dataset4 = GraphData(str(data_dir / "graph_features/benign_source/dataset4/cfg_embeddings"))

    # Benign Target datasets
    benign_target_dataset1 = GraphData(str(data_dir / "graph_features/benign_target/dataset1/cfg_embeddings"))
    benign_target_dataset2 = GraphData(str(data_dir / "graph_features/benign_target/dataset2/cfg_embeddings"))
    benign_target_dataset3 = GraphData(str(data_dir / "graph_features/benign_target/dataset3/cfg_embeddings"))
    benign_target_dataset4 = GraphData(str(data_dir / "graph_features/benign_target/dataset4/cfg_embeddings"))

    # Merge benign source datasets
    benign_source = merge_dataset(benign_source_dataset1, benign_source_dataset2)
    benign_source = merge_dataset(benign_source, benign_source_dataset3)
    benign_source = merge_dataset(benign_source, benign_source_dataset4)

    # Merge benign target datasets
    benign_target = merge_dataset(benign_target_dataset1, benign_target_dataset2)
    benign_target = merge_dataset(benign_target, benign_target_dataset3)
    benign_target = merge_dataset(benign_target, benign_target_dataset4)


    source_malware = merge_dataset(malware_march, malware_april)
    source_malware = merge_dataset(source_malware,  malware_may)

    target_malware_train = malware_sep
    target_malware_test = malware_oct


    source_normal =  benign_source
    target_normal = benign_target

    # convert it to binary labels
    target_malware_train = binary_label(target_malware_train, True)
    target_malware_test = binary_label(target_malware_test, True)
    source_malware  = binary_label(source_malware, True)
    target_normal = binary_label(target_normal, False)
    source_normal = binary_label(source_normal, False)


    source = merge_dataset(source_normal, source_malware)
    source_train, source_test = train_test_split(source, 0.75)


    target_normal_train, target_normal_test = train_test_split(target_normal, 0.5)
    target_train = merge_dataset(target_normal_train, target_malware_train)
    target_test = merge_dataset(target_normal_test, target_malware_test)


    ################################################################################
    # Config
    ################################################################################
    learning_rate_1 = 1e-3  # Learning rate
    learning_rate_2 = 0.001
    channels = 128  # Hidden units
    layers = 3  # GIN layers
    epochs = 20  # Number of training epochs
    batch_size = 16  # Batch size
    num_classes = 2

    print("Source malware size: {}".format(len(source_malware)))
    print("Source normal size: {}".format(len(source_normal)))
    print("Target malware train size: {}".format(len(target_malware_train)))
    print("Target malware test size: {}".format(len(target_malware_test)))
    print("Target normal size: {}".format(len(target_normal)))

    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))


        print("Source training set size is {}".format(len(source_train)))
        print("Target training set size is {}".format(len(target_train)))
        print("Target testing set size is {}".format(len(target_test)))



        loader_tr_source = DisjointLoader(source_train, batch_size=batch_size)
        loader_tr_target = DisjointLoader(target_train, batch_size=batch_size)
        loader_te = DisjointLoader(target_test, batch_size=batch_size)


        model = GIN0(channels, layers, num_classes)
        optimizer_1 = Adam(learning_rate_1)
        optimizer_2 = Adam(learning_rate_2)

        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc"])


        model.fit(loader_tr_source.load(), steps_per_epoch=loader_tr_source.steps_per_epoch, epochs=20)


        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 1

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False


        model.compile(optimizer=optimizer_2, loss="categorical_crossentropy", metrics=["acc",  MacroF1(num_classes)])

        model.fit(loader_tr_target.load(), steps_per_epoch=loader_tr_target.steps_per_epoch,epochs=20,
                validation_data=loader_te.load(), validation_steps=loader_te.steps_per_epoch
        )




        ################################################################################
        # Evaluate model
        ################################################################################
        print("Testing model")
        loss, acc, f1 = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
        print("Done. Test loss: {}. Test acc: {}. Test f1 {}".format(loss, acc, f1))
