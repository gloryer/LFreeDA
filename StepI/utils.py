import os
import numpy as np
import pandas as pd
from PIL import Image


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow.keras.backend as K



# Function to encode labels into one-hot format
def encode(label_y, classes):
    
    one_hot = np.zeros((len(label_y), classes))
    

    for idx, label in enumerate(label_y):
        # Convert 1-based label to 0-based index
        one_hot[idx, label] = 1

    return one_hot  






# Function to filter the dataset based on a specific label
def filter_label(df, y, label):
    #divide dataset based on the label[]
     
    output_df = df[np.where(y == label)[0]]
    
    res_df =  df[np.where(y != label)[0]]
    
    output_y = y[np.where(y == label)[0]]
    res_y = y[np.where(y != label)[0]]
    
    return  output_df, output_y, res_df, res_y

 

# Function to obtain labels from a DataFrame and a CSV file
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

# Function to change the attack label to a binary format
def change_attack_label(x):
    label = [1.]
    return label


# Function to load images and their corresponding labels from a directory
def load_image_malware(image_path, label_path):
    
    labels = pd.read_csv(label_path, header=0)
    
    x = []
    y = []
            
    
    
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            hash_id = filename.split(".")[0]
            if hash_id in labels['asm_id'].values: 
                f = os.path.join(image_path, filename)
                image = Image.open(f).convert('RGB')
                image = image.resize((56, 56), Image.ANTIALIAS)
                image = np.array(image, dtype=int)
                x.append(image)
                y.append(labels[labels["asm_id"] == hash_id]["Class"])

         
            
    x = np.asarray(x)
    y = np.asarray(y)
                       
    x = x.astype('float32') / 255.
    
    
        
    return x, y 


def load_image_malware_mb24(image_path, label_path):
    
    labels = pd.read_csv(label_path, header=0)
    
    x = []
    y = []
            
    
#     filenames_train = [img  for img in os.listdir(path_train) if img.endswith(".JPEG")]
#     filenames_test = [img for img in os.listdir(path_test) if img.endswith(".JPEG")]
    
    
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            hash_id = filename.split(".")[0]
            if hash_id in labels['malware SHA-256'].values: 
                f = os.path.join(image_path, filename)
                image = Image.open(f).convert('RGB')
                image = image.resize((56, 56), Image.ANTIALIAS)
                image = np.array(image, dtype=int)
                x.append(image)
                y.append(labels[labels["malware SHA-256"] == hash_id]["Label"])

         
            
    x = np.asarray(x)
    y = np.asarray(y)
                       
    x = x.astype('float32') / 255.
    
    
        
    return x, y 



def load_image_malware_md(image_path, label):

    
    x = []
    y = []
            
    
    
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            #hash_id = filename.split(".")[0]
            f = os.path.join(image_path, filename)
            image = Image.open(f).convert('RGB')
            image = image.resize((56, 56), Image.ANTIALIAS)
            image = np.array(image, dtype=int)
            x.append(image)
            y.append(label)

         
            
    x = np.asarray(x)
    y = np.asarray(y)
                       
    x = x.astype('float32') / 255.
    
    
        
    return x, y 




# Function to load images from a directory and resize them to a fixed size
def load_image_normal(directory_path):
    image_list = []
    image_size_limit = 178956970  # Maximum allowed pixels per image

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory_path, filename)
            try:
                Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                with Image.open(file_path) as img:
                    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                    # Check if the image size is within the allowed limit
                    if img.width * img.height <= image_size_limit:
                        img = img.convert('RGB')
                        img = img.resize((56, 56), Image.ANTIALIAS)
                        img_array = np.array(img, dtype=int)
                        image_list.append(img_array)
                    else:
                        print(f"Image {filename} exceeds the size limit of {image_size_limit} pixels and will be skipped.")
            except (Image.DecompressionBombError, OSError) as e:
                print(f"Error loading image {filename}: {e}")

    image_list = np.asarray(image_list)
    image_list = image_list.astype('float32') / 255.

    return image_list


def load_image_normal_md(directory_path, label):
    image_list = []
    image_size_limit = 178956970  # Maximum allowed pixels per image
    y = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory_path, filename)
            try:
                Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                with Image.open(file_path) as img:
                    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError check for large images
                    # Check if the image size is within the allowed limit
                    if img.width * img.height <= image_size_limit:
                        img = img.convert('RGB')
                        img = img.resize((56, 56), Image.ANTIALIAS)
                        img_array = np.array(img, dtype=int)
                        image_list.append(img_array)
                        y.append(label)
                    else:
                        print(f"Image {filename} exceeds the size limit of {image_size_limit} pixels and will be skipped.")
            except (Image.DecompressionBombError, OSError) as e:
                print(f"Error loading image {filename}: {e}")

    image_list = np.asarray(image_list)
    image_list = image_list.astype('float32') / 255.
    y = np.asarray(y)

    return image_list, y



# Custom metrics for model evaluation
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

