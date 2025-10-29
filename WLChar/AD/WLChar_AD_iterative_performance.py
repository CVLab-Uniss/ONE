from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import os
import math
import random
import torch
import time
from tqdm.notebook import tqdm
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

# Selezione del device
device_local = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Dispositivo in uso: {device_local}")

# Path del modello YOLO
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'yolo11m.pt')  

# Carica il modello YOLO (una sola volta)
model = YOLO(model_path)

# Funzioni di pre-processing
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def preProcImg(image):
    """Aggiunge padding per rendere l'immagine compatibile con YOLO (multipli di 32)."""
    width, height = image.size
    rw = 32 - (width % 32) if width % 32 != 0 else 0
    rh = 32 - (height % 32) if height % 32 != 0 else 0
    t = math.ceil(rh / 2)
    b = math.floor(rh / 2)
    l = math.ceil(rw / 2)
    r = math.floor(rw / 2)
    return add_margin(image, t, r, b, l, 0)


def process_one_video(video_path, frameSlot, yolo_conf, OUTPUT_RESULTS):
    """
    Esegue la logica su un singolo video e salva i risultati:
    <nome_video> - <secondo_rilevazione> - <YES|NO>
    """
    
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Errore nell'apertura del video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    video_name = os.path.basename(video_path)

    frame_idx = 0

    # --- Barra di progresso per i frame ---
    with tqdm(total=frame_count, desc=f"Analisi {video_name}", unit="frame") as pbar:
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frames = np.empty((frameSlot, frame_height, frame_width), dtype=np.uint8)

            valid_i = -1
            for i in range(frameSlot):
                ret, frame = cap.read()
                if not ret:
                    print(f"Errore nella lettura del frame {frame_idx + i}")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames[i] = gray
                valid_i = i

            if valid_i < 0:
                break

            # Calcolo della mediana dei frame
            median_frame = np.median(frames[:valid_i + 1], axis=0).astype(np.uint8)
            im = Image.fromarray(median_frame)
            im = preProcImg(im)
            im_width, im_height = im.size

            # Rilevamento oggetti con YOLO
            results = model.predict(im,
                                    imgsz=[im_height, im_width],
                                    augment=True,
                                    retina_masks=True,
                                    device=device_local,
                                    conf=yolo_conf,
                                    classes=[0, 1, 2, 3, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23],
                                    verbose=False)

            res = results[0].boxes.cls.cpu().numpy()
            detected = "YES" if res.size > 0 else "NO"

            det_frame = frame_idx + valid_i
            det_time_s = det_frame / frame_rate

            #print(f"Detected object: {detected} @ second {det_time_s:.2f}")

            # Scrittura risultato su file
            try:
                out_dir = os.path.dirname(OUTPUT_RESULTS)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                with open(OUTPUT_RESULTS, "a", encoding="utf-8") as f:
                    f.write(f"{video_name} {det_time_s:.3f} {detected}\n")
            except Exception as e:
                print(f"Impossibile scrivere su {OUTPUT_RESULTS}: {e}")

            frame_idx += frameSlot
            pbar.update(frameSlot)  # aggiorna barra dei frame

    cap.release()


def main():
    yolo_conf = 0.55
    fs_vector = [120, 240, 480, 600]    
    # Lista dei video disponibili
    video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_list:
        raise FileNotFoundError("Nessun file .mp4 trovato nella cartella specificata.")

    for frame_slot in fs_vector:
        print(f"frame_slot = {frame_slot}")

        # --- CONFIG: path output ---
        OUTPUT_CSV = "ad_iterations_timings.csv"
        OUTPUT_RESULTS = f"ad_results_{str(frame_slot)}_conf{str(yolo_conf).split('.')[-1]}.txt"
        
        # Pulizia iniziale del file risultati
        try:
            if os.path.exists(OUTPUT_RESULTS):
                os.remove(OUTPUT_RESULTS)
        except Exception as e:
            print(f"Attenzione: impossibile resettare {OUTPUT_RESULTS}: {e}")

        
        # --- Barra di progresso sui video ---
        for video_name in tqdm(video_list, desc="Analisi video", unit="video"):
            video_path = os.path.join(video_dir, video_name)
            process_one_video(video_path, frame_slot, yolo_conf, OUTPUT_RESULTS)


if __name__ == "__main__":
    main()
