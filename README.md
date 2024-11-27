
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

