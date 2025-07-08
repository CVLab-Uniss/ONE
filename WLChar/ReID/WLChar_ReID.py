# Questo script implementa un sistema di Re-Identification (ReID) per veicoli basato su visione artificiale, che integra modelli di deep learning e tecniche di ricerca approssimata di vettori. 
# L'architettura si fonda sull'utilizzo combinato di YOLO per il rilevamento di oggetti (bounding boxes di veicoli) e DinoV2 come estrattore di caratteristiche tramite una Vision Transformer.
# Il sistema è articolato nelle seguenti fasi principali:
# 1. Caricamento dei dati: vengono letti i metadati delle immagini da un file `.txt` contenente ID veicolo e ID della telecamera.
# 2. Definizione del modello: viene costruita una rete neurale personalizzata che estende DinoV2 con un classificatore a più layer, finalizzata all'estrazione e classificazione delle feature.
# 3. Pre-processing: si definisce una pipeline di trasformazioni per le immagini (ridimensionamento, normalizzazione).
# 4. Rilevamento veicoli con YOLO: si esegue il rilevamento degli oggetti (classi COCO relative ai veicoli) su un'immagine di input, salvando i ritagli contenenti veicoli.
# 5. Estrazione delle feature della query: a partire da un'immagine query, vengono calcolate le feature tramite il modello caricato e normalizzate per la successiva ricerca.
# 6. Costruzione della galleria: si itera su un set di immagini (escludendo la telecamera della query), si estraggono le feature e si indicizzano con FAISS.
# 7. Ricerca e confronto: viene effettuata una ricerca k-NN (k=5) per trovare i candidati più simili alla query. 
#    Il veicolo viene considerato correttamente identificato se la distanza del primo risultato è sufficientemente bassa e la varianza delle distanze è significativa.
# 8. Visualizzazione dei risultati: se il veicolo viene individuato, l'immagine corrispondente viene visualizzata insieme all'indicazione della telecamera e dell'ID veicolo.
# Il sistema è progettato per eseguire ReID su dataset realistici e può essere adattato per l'esecuzione su CPU o GPU. 

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

from ultralytics import YOLO
import cv2

filename = '../../Demo_ReID/test_3000_id.txt'
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = "cpu"
#device = "cuda"
#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("DEVICE used: ", device)

df = pd.read_csv(filename, sep=" ", header=None, names=["img", "v_id", "c_id"])
#camera_vect = [2, 30, 39, 102, 3, 172, 23, 137, 14, 79, 34, 78, 41, 51, 111, 110, 94, 139, 163, 122, 81]
camera_vect = [2]

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

# ********************* BBOX with YOLO *************************

# === PARAMETRI ===
input_image_path = "low_traffic.jpeg"         # Percorso immagine di input
output_folder = "cropped_objects"      # Cartella dove salvare i ritagli
#model_path = "yolov8m.pt"              # Modello YOLOv8
model_path = "yolo11m.pt" 
# === CREA CARTELLA DI OUTPUT SE NECESSARIO ===
os.makedirs(output_folder, exist_ok=True)

# === CARICA MODELLO YOLO ===
model_yolo = YOLO(model_path)

# === CLASSE VEICOLI (COCO classes) ===
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# === CARICA IMMAGINE ===
img_bgr = cv2.imread(input_image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Immagine non trovata: {input_image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# === PREDIZIONE ===
results = model_yolo.predict(source=img_rgb,
                        conf=0.35,
                        classes=vehicle_classes,
                        device=device,   # Cambia in "cuda" se hai una GPU
                        verbose=False)

# === ESTRAI BBOX E SALVA RITAGLI ===
boxes = results[0].boxes
if boxes is not None and boxes.xyxy is not None:
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = img_rgb[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped)
        save_path = os.path.join(output_folder, f"object_{i+1}.jpg")
        cropped_pil.save(save_path)
        print(f"Salvato: {save_path}")
else:
    print("Nessun veicolo rilevato.")

# ********************* LOAD MODEL AND CALCULATE QUERY FEATURES *******************

#model_name = 'test_mini_EE1.pth'
model_name = '../../Demo_ReID/test_full.pth'
model = torch.load(model_name).to(device)

# ********************* QUERY *******************

testImage = '../../Demo_ReID/Demo_cameras/14/002791.jpg' #query image for veichle id 256
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

# ********************* GALLERY *******************

# CALCULATE FEATURES FOR GALLERY IMAGES
# new vector without the query image (camera num 14)
#camera_vect = [2, 30, 39, 102, 3, 172, 23, 137, 14, 79, 34, 78, 41, 51, 111, 110, 94, 139, 163, 122, 81]
camera_vect = [2] 

datasetPath =  '../../Demo_ReID/Demo_cameras/'
print("Query image (vehicle id: ", img_id, ")")
display(testimg_or.resize((250,250)))

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
        display(image.resize((250,250)))
    else:
        bestIdx = indexes[0][0]
        image = Image.open(images[bestIdx])
        print("Query vehicle NOT found in camera num. ", cam)
        # display(image) #uncomment for debug