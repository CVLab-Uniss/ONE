
# Demo ReID 

This repository contains code and data for the task **Re-Identification (ReID)**.

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

## Installation Instructions

For installing Docker in the base system, follow step 1 in the README of the ONE project.

---

### **2. Create a Docker Container with CUDA**
To create a container with CUDA and access the Bash shell:
```bash
docker run --name=base-container -ti --runtime=nvidia --gpus all ubuntu /bin/bash
```

---

### **3. Install Python and Pip in the Container**

#### Install Python:
```bash

apt-get update
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

### **5. Install Required Python Packages for Demo_ReID**
Install the necessary dependencies for the Re-Identification task:
```bash
pip install pandas transformers faiss-cpu torch Pillow torchvision matplotlib
```

To copy the pretrained model on the container **run this command from the HOST system.**
```bash
docker cp test_full.pth base-container:/ONE/Demo_ReID
```

---

### **6. Save Changes to the Docker Container**
To save the current state of your container **run this command from the HOST system.**:
```bash
docker commit base-container one/reid-base:v1
```

---

### **7. Create an Auto-Start Script for the Demo**

#### Create a new script `DemoReID_start.sh`:
```bash
nano DemoReID_start.sh
```

#### Paste the following content:
```bash
#!/bin/sh
source "one-env/bin/activate"
cd /ONE/Demo_ReID/
python Demo_ReID.py
```
#### Make the script executable
```bash
chmod +x DemoReID_start.sh
```

#### Test the Script:
Deactivate the virtual environment, then run the script:
```bash
deactivate
source DemoReID_start.sh
```

#### Save the Changes:
```bash
docker commit test-gpu one/reid-base:v2
```

---

### **8. Run the Container with Auto-Start**
To run the container with the auto-start script:
```bash
docker run --name=DemoReID -ti --rm --runtime=nvidia --gpus all one/reid-base:v2 bash -c 'source DemoReID_start.sh'
```

---

### **9. Export the Docker Image**
To export the Docker image as a `.tar.gz` file:
```bash
docker save one/reid-base:v2 | gzip > one_reid-base_v2.tar.gz
```
or
```bash
docker image save one/reid-base:v2 -o one_reid-base_v2.tar.gz
```dokcer 

#### Move the Image to Google Drive:
Upload the image to the ONE project folder on Google Drive:
[Google Drive Folder](https://drive.google.com/drive/folders/1tlQE4pZ97qlIT6QgKOcm2Y70yzsFFmYG?usp=drive_link)

#### NB: move as soon as possible the file to another directory since it is not synchronized with git and fetching updates from server could delete this file.

---

### **10. Import the Docker Image**
To load the Docker image on a new system:
```bash
docker load < /path/to/one_reid-base_v2.tar.gz
```

---
