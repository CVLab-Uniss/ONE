### **1. Run the DemoAD Container in interactive mode**
To run the container with the auto-start script:
```bash
docker run --name=DemoReID -ti --runtime=nvidia --gpus all one/reid-base:v2 bash
```
#### **Update ONE directory from gitHub**
Run git pull to download the WLChar code:
```bash
cd /ONE/
git pull
```
#### **Install YOLO**
Activate the env and install with pip:
```bash
source "one-env/bin/activate"
pip install ultralytics
```
Install these libraries:
```bash
apt-get update && apt-get install -y libgl1
apt-get update && apt-get install -y libglib2.0-0
```

---

### **2. Change the Auto-Start Script for the WLChar**

#### Change the script `DemoAD_start.sh`:
```bash
cd ..
nano DemoReID_start.sh
```

#### Paste the following content:
```bash
#!/bin/sh
source "one-env/bin/activate"
cd /ONE/WLChar/ReID/
python WLChar_ReID.py
```

#### Test the Script:
Run the script:
```bash
source DemoReID_start.sh
```
#### Check the device in WLChar_AD.py:
```bash
nano /ONE/WLChar/ReID/WLChar_ReID.py
```
Select the device for the specific HW:
```bash
device_local = "cpu" #for the Edge
```
```bash
device_local = "cuda" #for the cloud
```

#### Save the Changes:
On the host machine run:
For the edge:
```bash
docker commit DemoReID one/wlchar_reid:cpu
```
For the cloud:
```bash
docker commit DemoReID one/wlchar_reid:gpu
```
---

### **3. Run the Container with Auto-Start**
To run the container with the auto-start script, selcting the image with tag "cpu" for the edge and the image with tag "gpu" for the cloud:
```bash
docker run --name=WLChar_ReID -ti --rm --runtime=nvidia --gpus all one/wlchar_reid:cpu bash -c 'source DemoReID_start.sh'
```
---
### **4. Push iamges on Docker Hub**
Login in Docker Hub:
```bash
sudo docker login
```
Create tags for the images to be pushed:
```bash
docker image tag one/wlchar_reid:gpu pietroruiu/one-project:wlchar_reid_gpu
docker image tag one/wlchar_reid:cpu pietroruiu/one-project:wlchar_reid_cpu
```
Push all the tags of the images for the repository 
```bash
sudo docker image push --all-tags pietroruiu/one-project

#### **Test the image**
```bash
docker run --name=WLChar_ReID -ti --rm --runtime=nvidia --gpus all pietroruiu/one-project:wlchar_reid_cpu bash -c 'source DemoReID_start.sh'
```
---

