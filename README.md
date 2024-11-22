
# ONE
**Repository for the ONE Project (Code and Data)**

This repository contains code and data for two tasks: **Re-identification (ReID)** and **Anomaly Detection (AD)**.

---

## Task: **Re-Identification (ReID)**
The **Demo_ReID** folder contains data for the ReID task, including the following files:

- **Demo_ReID.py**: Python script that runs the ReID demo.
- **city_map.pdf**: A slide with the map of the selected routes for the demo.
- **runDemo.ipynb**: Jupyter notebook for running the demo.
- **test_3000_id.txt**: Support file with image data used in the demo.
- **Demo_cameras/**: Folder containing the images for the demo, organized by camera.

### Required Model:
To run the Python code, download the fine-tuned model from the following location:
[Download Model](https://drive.google.com/file/d/1qpcHKMnlRv7NbKCro2ls5v2p0HqHitTm/view?usp=drive_link)

---

## Task: **Anomaly Detection (AD)**
The **Demo_AD** folder contains data for the Anomaly Detection task, including:

- **aic21-track4-train-data/**: Folder containing example videos.
- **AnomalyDect.ipynb**: Jupyter notebook for running the demo.
- **AnomalyDect.py**: Python script for executing the Anomaly Detection demo.
- **yolov8m.pt**: Model used for object detection.

---

## Installation Instructions

### **1. Install Docker on the Host**
Follow the instructions to install Docker:
- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

#### Add Docker's Official GPG Key:
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

#### Add Docker Repository to Apt Sources:
```bash
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### Add User to Docker Group (to avoid using `sudo`):
```bash
sudo usermod -aG docker ${USER}
su - ${USER}
```

#### Install GPU Support for Docker:
Follow these guides to enable GPU support:
- [SaturnCloud Guide](https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/)
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration)

Install NVIDIA Container Toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Running a Sample Container with CUDA:
Ensure GPU support is working by running:
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

---

### **2. Create a Docker Container with CUDA**
To create a container with CUDA and access the Bash shell:
```bash
docker run --name=base-container -ti --rm --runtime=nvidia --gpus all ubuntu /bin/bash
```

---

### **3. Install Python and Pip in the Container**

#### Install Python:
```bash
apt-get install python3
python3 --version
```

#### Install Virtual Environment:
```bash
apt-get install python3-venv
```

#### Create a Virtual Environment for ONE:
```bash
python3 -m venv one-env
```

#### Activate the Virtual Environment:
```bash
source "one-env/bin/activate"
```

#### Install Pip:
```bash
apt-get install python3-pip
pip --version
```

---

### **4. Clone the ONE Project Repository**
#### Install Git and Nano:
```bash
apt install git nano
```

#### Clone the Repository:
```bash
git clone https://github.com/CVLab-Uniss/ONE.git
```

#### Set Up GitHub Credentials:
You will need a GitHub username and password. Use a personal access token generated from your GitHub account:
- Go to **Settings** > **Developer Settings** > **Personal Access Tokens** > **Tokens (Classic)**.

---

### **5. Install Required Python Packages for Demo_AD**
Install the necessary dependencies for the Anomaly Detection task:
```bash
pip install ultralytics numpy pillow opencv-python opencv-python-headless
```

---

### **6. Save Changes to the Docker Container**
To save the current state of your container:
```bash
docker commit test-gpu one/ad-base:v1
```

---

### **7. Create an Auto-Start Script for the Demo**

#### Create a new script `DemoAD_start.sh`:
```bash
nano DemoAD_start.sh
```

#### Paste the following content:
```bash
#!/bin/sh
source "one-env/bin/activate"
cd /ONE/Demo_AD/
python AnomalyDect.py
```

#### Test the Script:
Deactivate the virtual environment, then run the script:
```bash
deactivate
source DemoAD_start.sh
```

#### Save the Changes:
```bash
docker commit test-gpu one/ad-base:v2
```

---

### **8. Run the Container with Auto-Start**
To run the container with the auto-start script:
```bash
docker run --name=DemoAD -ti --rm --runtime=nvidia --gpus all one/ad-base:v2 bash -c 'source DemoAD_start.sh'
```

---

### **9. Export the Docker Image**
To export the Docker image as a `.tar.gz` file:
```bash
docker save one/ad-base:v2 | gzip > one_ad-base_v2.tar.gz
```
or
```bash
docker image save one/ad-base:v2 -o one_ad-base_v2.tar.gz
```

#### Move the Image to Google Drive:
Upload the image to the **ONE** project folder on Google Drive:
[Google Drive Folder](https://drive.google.com/drive/folders/1tlQE4pZ97qlIT6QgKOcm2Y70yzsFFmYG?usp=drive_link)

---

### **10. Import the Docker Image**
To load the Docker image on a new system:
```bash
docker load < /path/to/one_ad-base_v2.tar.gz
```

---
