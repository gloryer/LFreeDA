
import numpy as np
from spektral.data import Dataset,  Graph
import scipy.sparse as sp
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()





class list_to_spektral_dataset(Dataset):
    def __init__(self, data, **kwargs):
        # Only needed transiently for read() (called by super().__init__()
        # below, which populates self.graphs from it) -- drop it right after
        # so the original list of Graph objects doesn't stay alive
        # duplicating the copies read() just made.
        self._source_data = data
        super().__init__(**kwargs)
        del self._source_data

    def read(self):
        return [Graph(x=graph.x, a=graph.a, y=graph.y) for graph in self._source_data]
    
    
    

# drop-in metric (same as I shared before)
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(
            name="cm", shape=(self.num_classes, self.num_classes),
            initializer="zeros", dtype=tf.float32
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        cm_batch = tf.math.confusion_matrix(
            y_true_labels, y_pred_labels,
            num_classes=self.num_classes, dtype=tf.float32
        )
        self.cm.assign_add(cm_batch)
    def result(self):
        TP = tf.linalg.tensor_diag_part(self.cm)
        FP = tf.reduce_sum(self.cm, axis=0) - TP
        FN = tf.reduce_sum(self.cm, axis=1) - TP
        precision = tf.math.divide_no_nan(TP, TP + FP)
        recall    = tf.math.divide_no_nan(TP, TP + FN)
        f1        = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)
    def reset_states(self):
        self.cm.assign(tf.zeros_like(self.cm))
    




def encode(dataset, classes):
    
    for g in dataset: 
        #print(g.y)
        y = np.zeros((classes,))
        #print(g.y)
        y[g.y] = 1
        g.y = y 
        
    return dataset 



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape



def merge_dataset(dataset1, dataset2): 
    # implemented fucntions in Dataset class of spektral graph 
    merged = dataset1. __add__(dataset2)
    return merged 


def binary_label(dataset, flag_attack):
    
    for g in dataset: 
        #g.y = np.pad(g.y, (0, 1), 'constant')
        if flag_attack: 
            y = np.zeros((2,))
            y[1] = 1
            g.y = y
        else:
            y = np.zeros((2,))
            y[0] = 1
            g.y = y
        
    return dataset
