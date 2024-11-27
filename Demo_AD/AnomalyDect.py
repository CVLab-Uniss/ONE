from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import os
import math

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

#load the model and processor
device_local = "cpu"
#device_local = "cuda"

model = YOLO('yolov8m.pt')

#lista delle classi
#print(model.names)

pth = "aic21-track4-train-data/"
#fname = "33.mp4" #Anomalia: Auto si ferma in corsia di ingresso
#fname = "34.mp4" #Nessuna anomalia
#fname = "49.mp4" #Anomalia: Auto si ferma in corsia di emergenza
#fname = "50.mp4" #Nessuna anomalia

# frame rate 30 frame/sec
frameSlot = 360 #numero di frame utilizzati per il calcolo del background
timeSlot = 3600 #numero di frame di intervallo fra un calcolo bg e l'altro (2 min di intervallo)
counter = timeSlot

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
    newsize = (416, 800)
    rw = 32 - (width%32) if width%32 != 0 else 0 
    rh = 32 - (height%32) if height%32 != 0 else 0
    t = math.ceil(rh/2) # arrotondamento intero superiore 
    b = math.floor(rh/2) # arrotondamento intero inferiore
    l = math.ceil(rw/2) 
    r = math.floor(rw/2)    
    image_res = add_margin(image, t, r, b, l, 0)
    
    return image_res


for fname in os.listdir(pth) :
    if '.mp4' in fname :
        print("Processing stream: ", fname)
        cap = cv2.VideoCapture(pth+fname)
        
        # Dimensioni dei frame (altezza e larghezza)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Numero totale di frame nel video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Crea una matrice tridimensionale vuota per contenere i frame
        frames = np.empty((frameSlot, frame_height, frame_width), dtype=np.uint8)
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print(f"End of frame.\n")
                break 
                  
            if (counter%timeSlot == 0):                
                for i in range(frameSlot):                       
                    ret, frame = cap.read()   
                    
                    if not ret:
                        print(f"Error reading the frame {i}.")
                        break                        
                        
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                median_frame = np.median(frames, axis=0).astype(np.uint8)
                counter = 0
        
                im = Image.fromarray(median_frame)
                im = preProcImg(im)
                frame_width, frame_height = im.size
                
                # list of Results objects
                results = model.predict(im, 
                                imgsz=[frame_height, frame_width], 
                                augment=True, 
                                retina_masks=True, 
                                device=device_local,
                                conf=0.35,
                                classes = [0, 1, 2, 3, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23],
                                show=False,
                                save=False,
                                save_txt=False,
                                verbose=False)  
                
                res = results[0].boxes.cls.cpu().numpy()
                if (res.size > 0):
                    print("Detected object:")
                    for t in range(res.size):
                        label = int(res[t])
                        print(model.names[label])
                else:
                    print("Detected object: None")
                
            counter = counter + 1
            
        # Rilascia la risorsa del video
        cap.release()
