---

### **8. Run the DemoAD Container in interactive mode**
To run the container with the auto-start script:
```bash
docker run --name=DemoAD -ti --runtime=nvidia --gpus all one/ad-base:v2 bash -c
```
---


### **7. Chenage the Auto-Start Script for the WLChar**

#### Change the script `DemoAD_start.sh`:
```bash
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
Deactivate the virtual environment, then run the script:
```bash
deactivate
source DemoAD_start.sh
```
#### Check the device in WLChar_AD.py:
```bash
nano WLChar_AD.py
```
Select the device for the specifi HW
- device_local = "cpu" for the Edge
- device_local = "cuda" for the cloud

#### Save the Changes:
For the edge:
```bash
docker commit test-gpu one/wlchar_ad:cpu
```
For the cloud:
```bash
docker commit test-gpu one/wlchar_ad:gpu
```
---
