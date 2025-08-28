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

video_name = random.choice(video_list)
video_path = os.path.join(video_dir, video_name)
print(f"Video selezionato: {video_name}")

# Parametri per l'elaborazione
frameSlot = 360   # numero di frame da usare per calcolare il background
timeSlot = 3600   # distanza tra due analisi successive

# Carica il modello YOLO
device_local = "cpu"
#device_local = "cuda"
#device_local = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#model = YOLO('yolov8m.pt')
model = YOLO('yolo11m.pt')
print(f"Dispositivo in uso: {device_local}")

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
steps = 360
total_duration = 12  # secondi
sleep_per_step = total_duration / steps

# Barra di progresso per notebook
for _ in tqdm(range(steps), desc="Acquiring 360 frames from camera (12 seconds)"):
    time.sleep(sleep_per_step)

# Apre il video selezionato
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Errore nell'apertura del video: {video_name}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f"Totale frame nel video: {frame_count}")

frame_idx = 0

while frame_idx + frameSlot <= frameSlot:
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
    print("Detected object: ")
    if res.size > 0:
        for t in res:
            label = int(t)
            print(f" - {model.names[label]}")
    else:
         print("None")

    frame_idx += timeSlot  # Avanza di 3600 frame (2 minuti)

cap.release()
