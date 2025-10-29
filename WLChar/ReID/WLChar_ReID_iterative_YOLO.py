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

###### ATTENZIONE ######
## IL NUOVO MODELLO DI DINO NON È COMPATIBILE CON IL CODICE. L'ERRORE E' IL SEGUENTE "TypeError: scaled_dot_product_attention(): argument 'dropout_p' must be float, not Dropout"
## COME WORKAROUND È STATO SOSTITUITO IL FILE ATTENTION.PY: IN PARTICOLARE LA FUNZIONE FORWARD() di seguito
# def forward(self, x: Tensor) -> Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
#         attn = q @ k.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
# LA FUNZIONE PIÙ RECENTE (NON FUNZIONANTE PER NOI) È PRESENTE IN ATTENTION_OLD.PY (~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/)


import os
import time
import torch
import faiss
import pandas as pd
import numpy as np
import statistics as st
from PIL import Image
from copy import deepcopy
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm.notebook import tqdm 
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import cv2
import matplotlib.pyplot as plt
import csv
import shutil
import random



# =================== CONFIG ===================
filename = '../../Demo_ReID/test_3000_id.txt'
model_path = '../../Demo_ReID/test_full.pth'
input_image_path = "low_traffic.jpeg"
output_folder = "cropped_objects"
datasetPath = '../../Demo_ReID/Demo_cameras/'
camera_vect = [200]  # cameras to scan in gallery

it_number = 300
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# =================== DEVICE ===================
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("DEVICE used: ", device)

# =================== LOAD DATA ===================
df = pd.read_csv(filename, sep=" ", header=None, names=["img", "v_id", "c_id"])

# =================== DEFINE TRANSFORMS ===================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =================== PREPARE DINO & MODEL CLASS ===================
# carichiamo dinov2
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# se nel checkpoint la classe custom era salvata come in precedenza, definiamola qui prima del torch.load
class_names = 30671
param = round(class_names / 4)

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vits14)
        self.classifier = nn.Sequential(
            nn.Linear(384, param),
            nn.ReLU(),
            nn.Linear(param, class_names)
        )

    def forward(self, x, return_embeddings=False):
        embeddings = self.transformer(x)
        if return_embeddings:
            return embeddings
        x = self.transformer.norm(embeddings)
        x = self.classifier(x)
        return x

# =============== LOAD MODEL (test_full.pth & YOLO) ===============
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
print("Modello .pth caricato correttamente.")

yolo_model_path = "yolo11m.pt"
model_yolo = YOLO(yolo_model_path)
print("Modello YOLO caricato correttamente.")


for num_cars in tqdm(range(1, 10), desc="Numero di veicoli (num_cars)", unit="gruppo"):
    OUTPUT_CSV = f"reid_iterations_timings_YOLO_{device}_{num_cars}vehic.csv"

    # **************BEGINNING OF THE ITERATIVE PROCEDURE***************
    for it in tqdm(range(it_number), desc=f"Iterazioni per {num_cars} veicoli", leave=False, unit="iter"):
    
        cartella_200 = os.path.join(datasetPath, str(camera_vect[0]))
        sorgente = os.path.join(datasetPath, "14")
    
        # 1. Elimina la cartella '200' se esiste
        if os.path.exists(cartella_200):
            shutil.rmtree(cartella_200)
            #print(f"Cartella esistente rimossa: {cartella_200}")
    
        # 2. Crea una nuova cartella '200'
        os.makedirs(cartella_200)
        #print(f"Nuova cartella creata: {cartella_200}")
    
        # 3. Ottiene la lista di tutti i file nella cartella sorgente
        tutti_file = [f for f in os.listdir(sorgente) if os.path.isfile(os.path.join(sorgente, f))]
    
        # 4. Seleziona casualmente num_cars file
        if num_cars > len(tutti_file):
            raise ValueError("num_cars supera il numero di file disponibili nella cartella sorgente.")
    
        file_scelti = random.sample(tutti_file, num_cars)
    
        # 5. Copia i file scelti nella nuova cartella '200'
        for nome_file in file_scelti:
            src = os.path.join(sorgente, nome_file)
            dst = os.path.join(cartella_200, nome_file)
            shutil.copy2(src, dst)
    
        #print(f"{num_cars} file copiati casualmente da {sorgente} a {cartella_200}.")
    
        
        
        #print(f"\nIterazione {it+1} iniziata.")
        start_time = time.time()
    
        # =================== DETECT VEHICLES WITH YOLO ===================
        os.makedirs(output_folder, exist_ok=True)
    
        img_bgr = cv2.imread(input_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Immagine non trovata: {input_image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        results = model_yolo.predict(
            source=img_rgb,
            conf=0.35,
            classes=vehicle_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
    
        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = img_rgb[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cropped)
                save_path = os.path.join(output_folder, f"object_{i+1}.jpg")
                cropped_pil.save(save_path)
                #print(f"Salvato: {save_path}")
        else:
            print("Nessun veicolo rilevato.")
    
        # =================== QUERY ===================
        testImage = '../../Demo_ReID/Demo_cameras/14/002791.jpg'
        df_filt = df.loc[df['img'].str.contains(os.path.basename(testImage))]
        img_id = df_filt['img'].values[0].split('/')[0]
    
        # Carica e trasforma immagine query
        testimg_or = Image.open(testImage).convert('RGB')
        testimg = transform(testimg_or).unsqueeze(0).to(device)
    
        # Estrai feature query
        with torch.no_grad():
            outputs = model(testimg, return_embeddings=True)
    
        vector = outputs.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vector)
    
        print("Query image (vehicle id: ", img_id, ")")
    
        # =================== ITERATIVE REID ===================
    
        # Crea CSV con intestazione se non esiste per gli output sul volume Docker
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["elapsed_time"])
    
        for cam in camera_vect:
            images = []
            cam_path = os.path.join(datasetPath, str(cam))
            for root, dirs, files in os.walk(cam_path):
                for file in files:
                    if file.endswith('.jpg'):
                        images.append(os.path.join(root, file))
    
    
            # Crea indice FAISS
            index = faiss.IndexFlatL2(384)
    
            # Funzione per aggiungere vettori normalizzati
            def add_vector_to_index(embedding, index):
                vec = embedding.detach().cpu().numpy().astype(np.float32)
                faiss.normalize_L2(vec)
                index.add(vec)
    
            # Estrai feature gallery e aggiungi a FAISS
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img_tensor, return_embeddings=True)
                add_vector_to_index(feat, index)
    
            # k-NN search
            distances, indexes = index.search(vector, 5)
            dist_vec = [distances[0, i] for i in range(len(indexes[0]))]
            bestIdx = indexes[0][0]
            best_image_path = images[bestIdx]
    
            if distances[0, 0] < 1.0 and st.variance(dist_vec) > 0.02:
                df_filt = df.loc[df['img'].str.contains(os.path.basename(best_image_path))]
                img_id_best = df_filt['img'].values[0].split('/')[0]
                print(f"Query vehicle found in camera {cam} (vehicle id: {img_id_best})")
            else:
                print(f"Query vehicle NOT found in camera {cam}")
    
        # =================== TIMINGS ===================
        elapsed_time = time.time() - start_time
    
    
        # Salva timing CSV
        try:
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([elapsed_time])
        except Exception as e:
            print(f"Impossibile scrivere su {OUTPUT_CSV}: {e}")
    
        #print(f"Iterazione {it+1} terminata. Tempo totale: {elapsed_time:.3f}s")