### **1. Run the DemoAD Container in interactive mode**
To run the container with the auto-start script:
```bash
docker run --name=DemoAD -ti --runtime=nvidia --gpus all one/ad-base:v2 bash -c
```
#### **Update ONE directory from gitHub**
Run git pull to download thw WLChar code:
```bash
cd /ONE/
git pull
```
---

### **2. Change the Auto-Start Script for the WLChar**

#### Change the script `DemoAD_start.sh`:
```bash
cd ..
nano DemoAD_start.sh
```

#### Paste the following content:
```bash
#!/bin/sh
source "one-env/bin/activate"
cd /ONE/WLChar/AD/
python WLChar_AD.py
```

#### Test the Script:
Run the script:
```bash
source DemoAD_start.sh
```
#### Check the device in WLChar_AD.py:
```bash
nano WLChar_AD.py
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
docker commit DemoAD one/wlchar_ad:cpu
```
For the cloud:
```bash
docker commit DemoAD one/wlchar_ad:gpu
```
---

### **3. Run the Container with Auto-Start**
To run the container with the auto-start script, selcting the image with tag "cpu" for the edge and the image with tag "gpu" for the cloud:
```bash
docker run --name=WLChar_AD -ti --rm --runtime=nvidia --gpus all one/wlchar_ad:cpu bash -c 'source DemoAD_start.sh'
```
---
### **4. Push iamges on Docker Hub**
Login in Docker Hub:
```bash
sudo docker login
```
Create tags for the images to be pushed:
```bash
docker image tag one/wlchar_ad:gpu pietroruiu/one-project:wlchar_ad_gpu
docker image tag one/wlchar_ad:cpu pietroruiu/one-project:wlchar_ad_cpu
```
Push all the tags of the images for the repository 
```bash
sudo docker image push --all-tags pietroruiu/one-project

### **Test the image**
```bash
docker run --name=WLChar_AD -ti --rm --runtime=nvidia --gpus all pietroruiu/one-project:wlchar_ad_cpu bash -c 'source DemoAD_start.sh'
```
