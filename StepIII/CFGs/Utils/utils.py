
import numpy as np
from spektral.data import Dataset,  Graph
import scipy.sparse as sp
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()





class list_to_spektral_dataset(Dataset):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(**kwargs)
        
    def read(self):
        #print(self.data[0])
        return [Graph(x=graph.x, a=graph.a, y=graph.y) for graph in self.data]
    
    
    
        
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    



def filter_label(dataset, label):
    #divide dataset based on the label[]
     
    output = []
    res = []
    
    for graph in dataset: 
        if graph.y in label:
            output.append(graph)
        else: 
            res.append(graph)
            
    output_dataset = list_to_spektral_dataset(output)
    res_dataset = list_to_spektral_dataset(res)
    return  output_dataset, res_dataset

def train_test_split(dataset, train_percentage):
    # Train/test split
    idxs = np.random.permutation(len(dataset)) 
    split = int(train_percentage * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    #print(bool(set(idx_tr) & set(idx_te)))
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]
    
    return dataset_tr, dataset_te


def train_test_split_2(dataset, train_percentage):
    # Train/test split
    idxs = np.arange(len(dataset)) 
    split = int(train_percentage * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    #print(bool(set(idx_tr) & set(idx_te)))
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]
    
    return dataset_tr, dataset_te



def subsample(dataset, size):
    # Train/test split
    idxs = np.random.permutation(len(dataset)) 
    split = int(size)
    idx_sample, idx_re = np.split(idxs, [split])
    dataset_sample, dataset_re= dataset[idx_sample], dataset[idx_re]
    
    return dataset_sample, dataset_re

def merge_dataset(normal, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7): 
    # implemented fucntions in Dataset class of spektral graph 
    merged = dataset1. __add__(dataset2)
    merged = merged. __add__(dataset3)
    merged = merged. __add__(dataset4)
    merged = merged. __add__(dataset5)
    merged = merged. __add__(dataset6)
    merged = merged. __add__(dataset7)
    merged = merged. __add__(normal)
    
    return merged 


def merge_dataset_mal(dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7): 
    # implemented fucntions in Dataset class of spektral graph 
    merged = dataset1. __add__(dataset2)
    merged = merged. __add__(dataset3)
    merged = merged. __add__(dataset4)
    merged = merged. __add__(dataset5)
    merged = merged. __add__(dataset6)
    merged = merged. __add__(dataset7)
    
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


def encode(dataset, classes):
    
    for g in dataset: 
        #print(g.y)
        y = np.zeros((classes,))
        #print(g.y)
        y[g.y] = 1
        g.y = y 
        
    return dataset 

def convert_label_integer_normal(dataset):
    for g in dataset: 
        #g.y = np.pad(g.y, (0, 1), 'constant')
        g.y = 0
        
    return dataset

def convert_label_integer(dataset):
    for g in dataset: 
        #g.y = np.pad(g.y, (0, 1), 'constant')
        g.y = np.where(g.y==1.)[0][0] 
        
    return dataset

def update_labels(dataset, path):
    
    
    labels = np.load(path)
    
    if len(labels)!=len(dataset):
        print("Label size {} does not match data size{}".format(len(labels),len(dataset)))
           
    i = 0 
    for g in dataset:
        g.y=labels[i]
        i+=1
        
    return dataset
    


def obtain_labels(dataset):
    
    y = []
    
    for g in dataset:
        y.append(g.y)
    
    return np.array(y)


def obtain_feature_matrix(dataset):

    graph_x = []
    

    
    for g in dataset:
        graph_x.append(np.sum(g.x, axis=0))
    return np.array(graph_x)



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
