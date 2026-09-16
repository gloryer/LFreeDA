import argparse
import time

from spektral.data import DisjointLoader
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from GraphMatching.graph_matching import load_matched_graphs
from Utils.utils import encode, MacroF1
from Lower_bound.model import GIN0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()
    start_time = time.time()

    ################################################################################
    # Config
    ################################################################################
    learning_rate_1 = 1e-3  # Learning rate
    learning_rate_2 = 0.001
    channels = 128  # Hidden units
    layers = 3  # GIN layers
    epochs = args.epochs
    batch_size = 16  # Batch size
    n_out = 2
    num_classes = 2

    #Load source train data
    path = '../../../data/stepII_constructed_datasets/mb24/aug/source_train.npz'
    source_train = load_matched_graphs(path, "source_path_train", "source_y_train","source_y_train")

    #Load target test data
    path = "../../../data/stepII_constructed_datasets/mb24/aug/target_test.npz"
    target_test = load_matched_graphs(path, "target_path_test", "target_y_test","target_y_test")

    source_train = encode(source_train , 2)
    target_test = encode(target_test , 2)

    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

        print("Source training set size is {}".format(len(source_train)))
        print("Target testing set size is {}".format(len(target_test)))

        loader_tr_source = DisjointLoader(source_train, batch_size=batch_size)
        loader_te = DisjointLoader(target_test, batch_size=batch_size)

        model = GIN0(channels, layers, n_out)
        optimizer_1 = Adam(learning_rate_1)

        model.compile(optimizer=optimizer_1, loss="categorical_crossentropy", metrics=["acc", MacroF1(num_classes)])

        model.fit(loader_tr_source.load(), steps_per_epoch=loader_tr_source.steps_per_epoch,epochs=epochs,
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



