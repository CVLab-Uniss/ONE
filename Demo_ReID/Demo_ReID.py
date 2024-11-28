import pandas as pd
import shutil

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
import faiss
import time
from torchvision import transforms 
from torch import nn, optim
from torchvision import datasets, transforms
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import statistics as st

filename = 'test_3000_id.txt'
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

df = pd.read_csv(filename, sep=" ", header=None, names=["img", "v_id", "c_id"])
camera_vect = [2, 30, 39, 102, 3, 172, 23, 137, 14, 79, 34, 78, 41, 51, 111, 110, 94, 139, 163, 122, 81]

class_names = 30671
param = round(class_names/4)

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vits14)
        self.classifier = nn.Sequential(nn.Linear(384, param), nn.ReLU(), nn.Linear(param, len(class_names)))

    def forward(self, x, return_embeddings=False):
        embeddings = self.transformer(x)
        
        if return_embeddings:
            # Se richiesto, restituisce solo gli embeddings
            return embeddings
        
        x = self.transformer.norm(embeddings)
        x = self.classifier(x)
        return x

#Define transformations for the dataset 
transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 

# LOAD MODEL AND CALCULATE QUERY FEATURES

#load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#model_name = 'test_mini_EE1.pth'
model_name = 'test_full.pth'
model = torch.load(model_name).to(device)

testImage = 'Demo_cameras/14/002791.jpg' #query image for veichle id 256
#testImage = 'Demo_cameras/14/001492.jpg' #query image for veichle id 145 (failed)
#testImage = 'Demo_cameras/14/004171.jpg' #query image for veichle id 408

df_filt = df.loc[df['img'].str.contains(testImage.split('/')[-1])]
img_id = df_filt['img'].values[0].split('/')[0]

#query image
testimg_or = Image.open(testImage).convert('RGB')
testimg = transform(testimg_or).unsqueeze(0).to(device)  

#Extract the features
with torch.no_grad():
        outputs = model(testimg, return_embeddings=True)

#Normalize the features before search
vector = outputs.detach().cpu().numpy()
vector = np.float32(vector)
faiss.normalize_L2(vector)


# CALCULATE FEATURES FOR GALLERY IMAGES
# new vector without the query image (camera num 14)
camera_vect = [2, 30, 39, 102, 3, 172, 23, 137, 79, 34, 78, 41, 51, 111, 110, 94, 139, 163, 122, 81]

print("DEVICE used: ", device)

datasetPath =  'Demo_cameras/'
print("Query image (vehicle id: ", img_id, ")")
#display(testimg_or)

for cam in camera_vect:
    #print("Camera id:", cam)
    #Populate the images variable with all the images in the dataset folder
    images = []

    for root, dirs, files in os.walk(datasetPath + str(cam)):
        for file in files:
            if file.endswith('jpg'):
                images.append(root  + '/'+ file)

    #Define a function that normalizes embeddings and add them to the index
    def add_vector_to_index(embedding, index):
        vector = embedding.detach().cpu().numpy()
        vector = np.float32(vector)
        #Normalize vector: important to avoid wrong results when searching
        faiss.normalize_L2(vector)
        index.add(vector)

    #Create Faiss index using FlatL2 type with 384 features
    index = faiss.IndexFlatL2(384) #small

    t0 = time.time()
    for image_path in images:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  
        with torch.no_grad():
            outputs = model(img, return_embeddings=True)
        add_vector_to_index(outputs, index)

    #print('Extraction done in :', time.time()-t0)
    
    distances, indexes = index.search(vector, 5)
    #print('distances:', distances, 'indexes:', indexes)
    
    dist_vec = []
    id = 0
    for i in indexes[0]:
        im = images[i]
        img_id = im.split('/')[-1]
        dist_vec.append(distances[0, id])
        id = id +1 
    
    if (distances[0, 0] < 1.0) and (st.variance(dist_vec) > 0.02):
        bestIdx = indexes[0][0]
        image = Image.open(images[bestIdx])
        df_filt = df.loc[df['img'].str.contains(images[bestIdx].split('/')[-1])]
        img_id = df_filt['img'].values[0].split('/')[0]
        print("Query vehicle found in camera num. ", cam, " (vehicle id:", img_id,")")
        #display(image)
    else:
        bestIdx = indexes[0][0]
        image = Image.open(images[bestIdx])
        print("Query vehicle NOT found in camera num. ", cam)
        # display(image) #uncomment for debug