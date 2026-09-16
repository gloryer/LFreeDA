from spektral.data import DisjointLoader
from tensorflow.keras import layers as keras_layers
from tensorflow.keras.optimizers import Adam


import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from GraphMatching.graph_matching import load_matched_graphs
from Utils.utils import encode, MacroF1
from AdvDA.model import AdvDA_GIN, GIN0

import argparse
import time
from sklearn.metrics import f1_score
import numpy as np




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()
    start_time = time.time()

    #Load source train data
    path = '../../../data/stepII_constructed_datasets/mb24/nov/source_train.npz'
    source_train = load_matched_graphs(path, "source_path_train", "source_y_train","source_y_train")



    #Load source test data
    path = "../../../data/stepII_constructed_datasets/mb24/nov/source_test.npz"
    source_test = load_matched_graphs(path, "source_path_test", "source_y_test","source_y_test")


    # Load selected target train data with pseudo-labels
    path = '../../../data/stepII_constructed_datasets/mb24/nov/target_train_filtered.npz'
    target_train_filtered = load_matched_graphs(path, "target_path_train_filtered", "target_pred_train_filtered","target_true_train_filtered")


    #Load target test data
    path = "../../../data/stepII_constructed_datasets/mb24/nov/target_test.npz"
    target_test = load_matched_graphs(path, "target_path_test", "target_y_test","target_y_test")




    source_train = encode(source_train, 2)
    target_train_filtered = encode(target_train_filtered, 2)
    target_test = encode(target_test, 2)


    ################################################################################
    # Config
    ################################################################################
    learning_rate_1 = 1e-3  # Learning rate
    learning_rate_2 = 0.001
    channels = 128  # Hidden units
    num_layers = 3  # GIN layers
    epochs = args.epochs
    batch_size = 16  # Batch size
    n_out = 2
    num_classes = 2


    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))


        print("Source training set size is {}".format(len(source_train)))
        print("Target training set size is {}".format(len(target_train_filtered)))
        print("Target testing set size is {}".format(len(target_test)))



        loader_tr_source = DisjointLoader(source_train, batch_size=batch_size)
        loader_tr_target = DisjointLoader(target_train_filtered, batch_size=batch_size)
        loader_te = DisjointLoader(target_test, batch_size=batch_size)


        model = GIN0(channels, num_layers, n_out)
        optimizer_1 = Adam(learning_rate_1)
        optimizer_2 = Adam(learning_rate_2)

        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc"])


        model.fit(loader_tr_source.load(), steps_per_epoch=loader_tr_source.steps_per_epoch, epochs=epochs)


        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 1

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False


        model.compile(optimizer=optimizer_2, loss="categorical_crossentropy", metrics=["acc", MacroF1(num_classes)])

        model.fit(loader_tr_target.load(), steps_per_epoch=loader_tr_target.steps_per_epoch, epochs=epochs,
                validation_data=loader_te.load(), validation_steps=loader_te.steps_per_epoch
        )




        ################################################################################
        # Evaluate model
        ################################################################################
        print("Testing model")
        loss, acc, f1 = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
        print("Done. Test loss: {}. Test acc: {}. Test f1 {}".format(loss, acc, f1))

        elapsed = time.time() - start_time
        print("Total runtime: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))


