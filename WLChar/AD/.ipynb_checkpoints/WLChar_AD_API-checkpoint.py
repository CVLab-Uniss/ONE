# Questo script Python esegue il rilevamento automatico di oggetti in un video selezionato casualmente da una cartella. 
# Utilizza il modello YOLO (You Only Look Once) per identificare veicoli e altri oggetti di interesse in ambiente stradale.
# ### Funzionamento:
# 1. Selezione casuale del video dalla directory specificata.
# 2. Seleziona i primi 3600 frame (equivalente a 2 minuti a 30 fps), il codice:
#  * legge un blocco di 360 frame consecutivi;
#  * calcola il frame mediano (background statico);
#  * esegue il rilevamento oggetti sul frame mediano tramite YOLO.
# 3. Gli oggetti rilevati vengono stampati a video con le relative classi (es. "car", "truck").

from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import os
import math
import random
import torch
import time
from tqdm import tqdm

import requests
import json
import urllib3

#https://github.com/ultralytics/ultralytics
#https://docs.ultralytics.com/modes/predict/#inference-arguments

'''
  2  587 894 Auto ferma in corsia di emergenza
  9    0 287 Alta densità di traffico? Cambio prospettiva
 11    0 888 Coda
 14  475 600 Coda. Cambio prospettiva.
 33    0 90  Auto ferma in corsia emergenza
 33  165 894 Auto si ferma in corsia di ingresso
 35  106 185 Camion fermo in basso a sinistra
 49  422 894 Auto si ferma in corsia emergenza
 51  431 891 Auto si ferma in corsia emergenza. Cambio di inquadratura
 63   87 853 Camion si ferma in corsia di emer. Nel finale cambia inquadratura
 72   87 894 Auto si ferma in corsia emergenza
 73  155 894 Camion si ferma in corsia di emergenza
 74  293 894 Auto si ferma in corsia emergenza 
 83  540 892 Auto si ferma in corsia emergenza. Nevica
 91  602 900 Auto fa testa coda.
 93    0 892 Auto ferma in corsia emergenza
 95   38 894 Incidente. Macchina si ribalta.
 97    0 890 Auto ferme in corsia emergenza
 '''

# Percorso della cartella contenente i video
video_dir = "../../Demo_AD/aic21-track4-train-data/"

# Seleziona un file video casuale dalla cartella
video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
if not video_list:
    raise FileNotFoundError("Nessun file .mp4 trovato nella cartella specificata.")

# Parametri per l'elaborazione
frameSlot = 360   # numero di frame da usare per calcolare il background
timeSlot = 3600   # distanza tra due analisi successive

# Carica il modello YOLO
device_local = "cpu"
#device_local = "cuda"
#device_local = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model_name = 'yolo11m.pt'
#model = YOLO('yolov8m.pt')
model = YOLO(model_name)
print(f"Local device in use: {device_local}")
print(f"Object detection model in use: {model_name}")
print("\n")

print("********** START RE-IDENTIFICATION TASK **********")

# In caso di anomalia invia un messaggio all'end-point dello slave node
def report_anomaly(object_type):
    # URL di destinazione
    url = "https://flask-app-aks-nodepool1-17379992-vmss000014-service.4.232.16.189.nip.io:443/report_anomaly"
    anomaly_type = object_type
    
    # Corpo della richiesta (payload)
    data = {
        "slave_node_ip": "flask-app-aks-nodepool1-17379992-vmss0000014-service",
        "anomaly_type": object_type,
        "timestamp": "2025-06-03 15:25:12",
        "vehicle_brand": "VW",
        "vehicle_model": "GOLF5",
        "vehicle_color": "white "
    }
    
    # Token di sicurezza (sostituisci con il tuo reale token)
    security_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ0cmFmZmljLW1hc3Rlci1ub2RlIiwic3ViIjoiMTMyLjE3Ny40OC4xMTMiLCJhdWQiOiJmbGFzay1hcHAtYWtzLW5vZGVwb29sMS0xNzM3OTk5Mi12bXNzMDAwMDE0LXNlcnZpY2UuNC4yMzIuMTYuMTg5Lm5pcC5pbyIsImlhdCI6MTc1NDA0NjQxNywiZXhwIjoxNzYyNjg2NDE3LCJzY29wZSI6ImNhbWVyYV9hcGkifQ.ylULJvR99_kNe9cepx81NPLGbc6y4lrT2sisaMRRtmshC1aoGJR54iheHrQKyHfG1R6HvvlK773N1S0Ozg4x15Ti3czAsteDxn9a2gqAJZw2PDQmdMfrjOoNDgzeikmrmyOaZyf9ZvoMHUM854OgugfXkKPeY6vGwdJv8xTDa6eANIUFlc2OXeeveo5VNYwrxxRTd7GTD0_xXpStQsLSIyN2PuGKrafuUqEm69MQmMWAlBIetnHG7A1GNFsT6GKvNhQyI6lUabiJuQTDrmZbVzbCggbs0hZTBLcMYNtkh5SkaEISRPartAFAqz3HOL5LM57C2oF9uB7X8q2uVzN6rA"
    
    # Headers con token di autorizzazione
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {security_token}"
    }
    
    # Disabilita i warning per certificati self-signed (facoltativo ma utile se il certificato HTTPS non è valido)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Invia la richiesta POST
    response = requests.post(url, data=json.dumps(data), headers=headers, verify=False)
    
    # Stampa la risposta del server
    print(f"Status code: {response.status_code}")
    print("Response body:", response.text)
    return response

# Funzioni di pre-processing
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# funzione per aggiungere il padding e rendere l'immagine compatibile con l'input del modello.
# Il modello accetta in ingresso immagini con H e W multipli di 32.

def preProcImg(image):
    width, height = image.size
    rw = 32 - (width % 32) if width % 32 != 0 else 0
    rh = 32 - (height % 32) if height % 32 != 0 else 0
    t = math.ceil(rh / 2)
    b = math.floor(rh / 2)
    l = math.ceil(rw / 2)
    r = math.floor(rw / 2)
    return add_margin(image, t, r, b, l, 0)


# Parametri
total_duration = 12  # secondi
sleep_per_step = total_duration / frameSlot

while True:
    video_name = random.choice(video_list)
    video_path = os.path.join(video_dir, video_name)
    #print(f"Video selezionato: {video_name}")
    print("Starting acquisition for background estimation")

    # Barra di progresso per notebook
    for _ in tqdm(range(frameSlot), desc="Acquiring 360 frames from camera (12 seconds)"):
        time.sleep(sleep_per_step)
    # Apre il video selezionato
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    if not cap.isOpened():
        raise IOError(f"Errore nell'apertura del video: {video_name}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Totale frame nel video: {frame_count}")

    #adding an offset for variability
    max_offset = frame_count - 3690
    if max_offset <= 0:
        raise ValueError("Il video è troppo corto per sottrarre 3690 frame.")
    
    # genera offset casuale
    offset = random.randint(0, max_offset)
    frame_idx = offset
    #print(offset)
    
    while frame_idx + frameSlot <= frameSlot + offset:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
        frames = np.empty((frameSlot, frame_height, frame_width), dtype=np.uint8)
    
        for i in range(frameSlot):
            ret, frame = cap.read()
            if not ret:
                print(f"Errore nella lettura del frame {frame_idx + i}")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i] = gray
    
        # Calcolo della mediana dei frame
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        im = Image.fromarray(median_frame)
        im = preProcImg(im)
        im_width, im_height = im.size
    
        # Rilevamento oggetti con YOLO
        results = model.predict(im,
                                imgsz=[im_height, im_width],
                                augment=True,
                                retina_masks=True,
                                device=device_local,
                                conf=0.35,
                                classes=[0, 1, 2, 3, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23],
                                verbose=False)
    
        res = results[0].boxes.cls.cpu().numpy()
        # DEBUG: print image with bbox
        #img = results[0].plot()
        #cv2.imwrite("test.jpg", img)
        if res.size > 0:
            for t in res:
                label = int(t)
                print(f"New object detected: {model.names[label]}")
                report_anomaly(label)
        else:
             print("No object detected")
    
        frame_idx += timeSlot  # Avanza di 3600 frame (2 minuti)
    
    cap.release()
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Parametri
    total_duration = 120 - elapsed  # secondi
    if total_duration < 0 :
        sleep_per_step = 0
    else :
        sleep_per_step = total_duration / timeSlot
    
    # Barra di progresso per notebook
    # for _ in tqdm(range(timeSlot), desc=f"Waiting {120 - elapsed} seconds for a new detection"):
    #     time.sleep(sleep_per_step)

