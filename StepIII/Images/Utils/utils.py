import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def encode(label_y, classes):
    
    one_hot = np.zeros((len(label_y), classes))
    

    for idx, label in enumerate(label_y):
        # Convert 1-based label to 0-based index
        one_hot[idx, label] = 1

    return one_hot  




def filter_label(df, y, label):
    #divide dataset based on the label[]
     
    output_df = df[np.where(y == label)[0]]
    
    res_df =  df[np.where(y != label)[0]]
    
    output_y = y[np.where(y == label)[0]]
    res_y = y[np.where(y != label)[0]]
    
    return  output_df, output_y, res_df, res_y

 


def obtain_labels(df, label_path):
    
    labels = pd.read_csv(label_path, header=0)
    
    y = []
    x = []
    
    for index, row in df.iterrows():
        #print(row["asm_id"].split(".")[0])
        hash_id = row["asm_id"].split(".")[0]
        if hash_id in labels['asm_id'].values: 
            row = row.drop("asm_id")
            x.append(row)
            y.append(labels[labels["asm_id"] == hash_id]["Class"])
            

    
    return np.array(x), np.array(y)


def change_attack_label(x):
    label = [1.]
    return label



def load_image(image_path, labels):

    

    
    x = []
    y = []
    
    
    
    for p, l in zip(image_path, labels):
        if p.endswith(".png"):
            #hash_id = filename.split(".")[0]

            image = Image.open("../" + p).convert('RGB')
            image = image.resize((56, 56), Image.ANTIALIAS)
            image = np.array(image, dtype=int)
            x.append(image)
            y.append(l) 

         
            
    x = np.asarray(x)
    y = np.asarray(y)
    
                       
    x = x.astype('float32') / 255.
    
    
        
    return x, y




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

