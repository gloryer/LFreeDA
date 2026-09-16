import argparse
import time
from spektral.data import DisjointLoader
from tensorflow.keras import layers


import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
project_root = script_path.parent.parent
sys.path.append(str(script_path))
from GraphMatching.graph_matching import load_matched_graphs
from Utils.utils import encode
from AdvDA.model import AdvDA_GIN, GIN0

from sklearn.metrics import f1_score
import numpy as np




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Number of training epochs (default: 60)")
    args = parser.parse_args()
    start_time = time.time()

    #Load source train data
    path = str(project_root / 'data/stepII_constructed_datasets/mb24/aug/source_train.npz')
    source_train = load_matched_graphs(path, "source_path_train", "source_y_train","source_y_train")



    # Load selected target train data with pseudo-labels
    path = str(project_root / 'data/stepII_constructed_datasets/mb24/aug/target_train_filtered.npz')
    target_train_filtered = load_matched_graphs(path, "target_path_train_filtered", "target_pred_train_filtered","target_true_train_filtered")


    #Load target test data
    path = str(project_root / 'data/stepII_constructed_datasets/mb24/aug/target_test.npz')
    target_test = load_matched_graphs(path, "target_path_test", "target_y_test","target_y_test")




    source_train = encode(source_train , 2)
    target_train_filtered = encode(target_train_filtered , 2)
    target_test = encode(target_test , 2)

    ################################################################################
    # Config
    ################################################################################
    channels = 128  # Hidden units
    layers = 3  # GIN layers
    epochs = args.epochs
    batch_size = 16  # Batch size
    n_out = 2

    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))

        print("Source training set size is {}".format(len(source_train)))
        print("Target training set size is {}".format(len(target_train_filtered)))
        print("Target testing set size is {}".format(len(target_test)))

        loader_source_tr = DisjointLoader(source_train, batch_size=batch_size, epochs = epochs, shuffle = True)

        loader_target_tr = DisjointLoader(target_train_filtered, batch_size=batch_size, epochs = None, shuffle = True)
        loader_target_te = DisjointLoader(target_test, batch_size=batch_size)

        # build model
        GIN = GIN0(channels, layers)

        model = AdvDA_GIN(loader_source_tr, loader_target_tr, loader_target_te, GIN, n_out, epochs=epochs)

        G, C = model.train()

        ################################################################################
        # Evaluate model
        ################################################################################
        all_preds, all_trues = [], []

        for step in range(loader_target_te.steps_per_epoch):
            inputs, target = loader_target_te.__next__()
            pred = C(G(inputs, training=False), training=False)

            all_trues.append(np.argmax(target, axis=1))
            all_preds.append(np.argmax(pred, axis=1))

        all_trues = np.concatenate(all_trues)
        all_preds = np.concatenate(all_preds)

        f1_weighted = f1_score(all_trues, all_preds, average='weighted')
        f1_macro    = f1_score(all_trues, all_preds, average='macro')
        print("Done. Test weighted F1: {:.4f}, macro F1: {:.4f}".format(f1_weighted, f1_macro))

        elapsed = time.time() - start_time
        print("Total runtime: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))