# ONE
Repository di codice e dati per il progetto ONE

Nella cartella Demo_ReID sono presenti i dati per il task di re-identification
I file sono i seguenti:
- Demo_ReID.py - codice python che esegue la demo del task di re-identification
- city_map.pdf - slide con la mappa dei percorsi selezionati per la demo
- runDemo.ipynb - codice per eseguire la demo su un notebook Jupyter
- test_3000_id.txt - file di supporto con i dati delle immagini usate per la demo
- Demo_cameras - cartella contenente le immagini scelte per la demo, suddivise per videocamera
Per eseguire il codice python è necessario scaricare il modello finetunato dalla seguente location
https://drive.google.com/file/d/1qpcHKMnlRv7NbKCro2ls5v2p0HqHitTm/view?usp=drive_link 

Nella cartella Demo_AD sono presenti i dati per il task di Anomaly Detection
I file sono i seguenti:
- aic21-track4-train-data - cartella con i video di esempio
- AnomalyDect.ipynb - codice per eseguire la demo su un notebook Jupyter
- AnomalyDect.py - codice python che esegue la demo del task di Anomaly Detection
- yolov8m.pt - modello usato per la detection

******************* INSTALLAZIONE DOCKER sull'HOST ******************* 
# Installare docker
# Fonte: https://docs.docker.com/engine/install/ubuntu/ 

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# aggiungere l'utente al gruppo docker per evitare di dover usare "sudo"
# Fonte: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
sudo usermod -aG docker ${USER}
su - ${USER}

# installare il supporto per le GPU per Docker (nella macchina host)
# Fonte: https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/
# Fonte: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Running a Sample Container with CUD: Your output should resemble the nvidia-smi output
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi 

******************* CREAZIONE DEL CONTAINER *******************

# Creare un container con CUDA e la bash
# eseguire nell'host
docker run --name=base-container -ti --rm --runtime=nvidia --gpus all ubuntu /bin/bash

# ******************* installazione python e pip ***********************

# python
apt-get install python3
python3 --version
# installare virtual environment
apt-get install python3-venv
# creare un vemv di ONE
python3 -m venv one-env
# attivare il venv
source one-env/bin/activate
# installare pip
apt-get install python3-pip
pip --version

# ******************* clone del porgetto ONE ***********************

# installare git e nano
apt install git nano
# clonare il progetto dal repo di UNISS
git clone https://github.com/CVLab-Uniss/ONE.git
# inserire username e password 
# utilizzare un utente associato al progetto
# per la password utilizzare un token generato dal proprio account di git hub
# settings -> development settings -> personal access token -> tokens (classic)

# installazione requirements per Demo_AD
pip install ultralytics numpy pillow opencv-python opencv-python-headless 

# salvataggio delle modifiche apportate al container 
# eseguire nell'host
docker commit test-gpu one/ad-base:v1

# ******************* Creazione script per avvio automatico della demo *******************

nano DemoAD_start.sh
# copiare il seguente testo
#!/bin/sh
source "one-env/bin/activate"
cd /ONE/Demo_AD/
python AnomalyDect.py
# test dello script
# uscire dall venv e lanciare lo script
deactivate
source DemoAD_start.sh

# salvataggio delle modifiche apportate al container 
# eseguire nell'host
docker commit test-gpu one/ad-base:v2

# ******************* Esecuzione del container con avvio automatico *******************

docker run --name=DemoAD -ti --rm --runtime=nvidia --gpus all one/ad-base:v2 bash -c 'source DemoAD_start.sh'

# ******************* Export dell'immagine ***************************
docker save one/ad-base:v2 | gzip > one_ad-base_v2.tar.gz
(oppure) docker image save one/ad-base:v2 -o one_ad-base_v2.tar.gz
# spostare l'immagine nella cartella del progetto su gDrive
# Progetto ONE -> Deliverables -> Sviluppo -> Docker_img
# https://drive.google.com/drive/folders/1tlQE4pZ97qlIT6QgKOcm2Y70yzsFFmYG?usp=drive_link 

# ******************* Import dell'immagine ***************************
docker load < /path/to/one_ad-base_v2.tar.gz



