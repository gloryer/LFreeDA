from spektral.data import DisjointLoader
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from GraphMatching.mb24 import load_matched_graphs
from Utils.utils import encode, f1_m
from AdvDA.model import AdvDA_GIN, GIN0

from sklearn.metrics import f1_score
import numpy as np




if __name__ == "__main__":

    

    #Load source train data
    path = '../../../data/stepII_constructed_datasets/mb24/july_task/source_train.npz'
    source_train = load_matched_graphs(path, "source_path_train", "source_y_train","source_y_train")



    #Load source test data
    path = "../../../data/stepII_constructed_datasets/mb24/july_task/source_test.npz"
    source_test = load_matched_graphs(path, "source_path_test", "source_y_test","source_y_test")
    

    # Load selected target train data with pseudo-labels
    path = '../../../data/stepII_constructed_datasets/mb24/july_task/target_train_filtered.npz'
    target_train_filtered = load_matched_graphs(path, "target_path_train_filtered", "target_pred_train_filtered","target_true_train_filtered")


    #Load target test data
    path = "../../../data/stepII_constructed_datasets/mb24/july_task/target_test.npz"
    target_test = load_matched_graphs(path, "target_path_test", "target_y_test","target_y_test")




    source_train = encode(source_train , 2)
    target_train_filtered = encode(target_train_filtered , 2)
    target_test = encode(target_test , 2)



    channels = 128  # Hidden units
    layers = 3  # GIN layers
    epochs = 50  # Number of training epochs
    batch_size = 16  # Batch size
    n_out = 2


    for i in range(1):
        print("--------------------------{} run-------------------------".format(i))


        print("Source training set size is {}".format(len(source_train)))
        print("Target training set size is {}".format(len(target_train_filtered)))
        print("Target testing set size is {}".format(len(target_test)))




        loader_source_tr = DisjointLoader(source_train, batch_size=batch_size, epochs = epochs, shuffle = True)



        loader_target_tr = DisjointLoader(target_train_filtered, batch_size=batch_size, epochs = epochs, shuffle = True)
        loader_target_te = DisjointLoader(target_test, batch_size=batch_size)

        # build model
        GIN = GIN0(channels, layers)


        model =AdvDA_GIN(loader_source_tr, loader_target_tr, loader_target_te, GIN, n_out, epochs=50)


        G, C = model.train()


        ################################################################################
        # Evaluate model
        ################################################################################
        results = []
        step = 0

        while step < loader_target_te.steps_per_epoch:      
            step += 1
            inputs, target = loader_target_te.__next__()
            pred = C(G(inputs, training=False), training=False)
            results.append(
                (
                f1_score(np.argmax(target, axis=1), np.argmax(pred, axis=1), average='weighted')
                )
            )
        print("Done. Test f1: {}".format(np.mean(results, 0)))




      
    