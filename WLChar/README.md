
# WLChar

This repository contains code and data for running single instances of two tasks: **Re-identification (ReID)** and **Anomaly Detection (AD)**.
Single istances means a basic pipeline (input/output) of each of the two tasks. 

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
This Python script performs automatic object detection on a video randomly selected from a folder.
It uses the YOLO (You Only Look Once) model to identify vehicles and other objects of interest in a road environment.
### Functionality:

1. Random selection of a video from the specified directory.
2. The script processes the first 3600 frames (equivalent to 2 minutes at 30 fps) by:

   * Reading a block of 360 consecutive frames;
   * Computing the median frame (representing the static background);
   * Performing object detection on the median frame using YOLO.
3. The detected objects are displayed on the screen along with their corresponding classes (e.g., "car", "truck").
