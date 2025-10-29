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
import argparse

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

# --- CONFIG: path output CSV (scrive su /data, utile quando facciamo mount -v volume_test:/data) ---
OUTPUT_CSV = "ad_iterations_timings.csv"
# -----------------------------------------------------------------------------------------------

# Parametri per l'elaborazione
frameSlot = 360   # numero di frame da usare per calcolare il background
timeSlot = 3600   # distanza tra due analisi successive
total_duration = 12  # secondi
sleep_per_step = total_duration / frameSlot

# Selezione del device
#device_local = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device_local = "cpu"

# Path del modello yolo già scaricato
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'yolo11m.pt')  
print(f"Dispositivo in uso: {device_local}")

# Carica il modello YOLO (caricato una sola volta)
model = YOLO(model_path)

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


#------------da qui in poi bisogna iterare
def process_one_video(video_path):
    """
    Esegue la logica originale su un singolo video:
    - apre il video
    - genera offset casuale e controlla max_offset
    - legge frameSlot frame, calcola mediana, esegue predict YOLO
    - simula i progressi come nell'originale
    """

    # Barra di progresso per notebook (simula l'acquisizione iniziale di 360 frame)
    # for _ in tqdm(range(frameSlot), desc="Acquiring 360 frames from camera (12 seconds)"):
    #     time.sleep(sleep_per_step)

    # Apre il video selezionato
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Errore nell'apertura del video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    #adding an offset for variability
    max_offset = frame_count - 3690
    if max_offset <= 0:
        cap.release()
        raise ValueError("Il video è troppo corto per sottrarre 3690 frame.")

    # genera offset casuale
    offset = random.randint(0, max_offset)
    frame_idx = offset

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
        print("Detected object: ")
        if res.size > 0:
            for t in res:
                label = int(t)
                print(f" - {model.names[label]}")
        else:
            print("None")

        frame_idx += timeSlot  # Avanza di 3600 frame (2 minuti)

    cap.release()
    end_time = time.time()
    elapsed = end_time - start_time

    # Parametri
    total_duration_iter = 120 - elapsed  # secondi
    if total_duration_iter < 0 :
        sleep_per_step_iter = 0
    else :
        sleep_per_step_iter = total_duration_iter / timeSlot

    # Barra di progresso per notebook (simula il resto del tempo in cui il video "scorre")
    # for _ in tqdm(range(timeSlot), desc=f"Acquiring {timeSlot} frames from camera ({120 - elapsed} seconds)"):
    #     time.sleep(sleep_per_step_iter)


def main():
    # Legge la lista dei video disponibili
    video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_list:
        raise FileNotFoundError("Nessun file .mp4 trovato nella cartella specificata.")


    executed = 0
    number = 30
    for i in range(number):
        # Tempo di inizio della iterazione
        iteration_start_time = time.time()
        # Ogni iterazione prende un video random nuovo (o lo stesso se scelto casualmente)
        video_name = random.choice(video_list)
        video_path = os.path.join(video_dir, video_name)
        print(f"\n[Iterazione {i+1}] Video selezionato: {video_name}")

        # Esegue la logica completa per il video selezionato
        process_one_video(video_path)
    
        # Calcoliamo la durata dell'iterazione
        iteration_end_time = time.time()
        it_duration = iteration_end_time - iteration_start_time
        
        # --- Scriviamo il valore del tempo di iterazione nel CSV (una sola colonna, una riga per iterazione, nient'altro)
        try:
            out_dir = os.path.dirname(OUTPUT_CSV)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(OUTPUT_CSV, "a", newline="") as f:
                f.write(f"{it_duration}\n")
        except Exception as e:
            print(f"Impossibile scrivere su {OUTPUT_CSV}: {e}")
            
        executed += 1

    # Fine del ciclo for
    print(f"\nIterazioni richieste: {number}. Iterazioni eseguite: {executed}.")


if __name__ == "__main__":
    main()