
# WLChar

This repository provides the code and data necessary to run basic implementations of two computer vision tasks: Re-Identification (ReID) and Anomaly Detection (AD).
By "basic implementations" we refer to simplified, standalone pipelines for each task, where a standard input the input is processed through the entire processing pipeline to produce the output. 
These implementations are intended to demonstrate the essential functioning of ReID and AD, in order to test the consumption of the basic pipeline.
The purpose of these code is to 

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

---

## Task: **Re-Identification (ReID)**

This script implements a vehicle Re-Identification (ReID) system based on computer vision, integrating deep learning models and approximate vector search techniques.
The architecture combines YOLO for object detection (vehicle bounding boxes) and DinoV2 as a feature extractor via a Vision Transformer.
### Functionality:

1. **Data loading**: Image metadata are read from a `.txt` file containing vehicle IDs and camera IDs.
2. **Model definition**: A custom neural network is built by extending DinoV2 with a multi-layer classifier, aimed at feature extraction and classification.
3. **Pre-processing**: An image transformation pipeline is defined (resizing, normalization).
4. **Vehicle detection with YOLO**: Object detection is performed on an input image (focused on COCO classes related to vehicles), and vehicle-containing crops are saved.
5. **Query feature extraction**: Features are computed from a query image using the loaded model and normalized for subsequent retrieval.
6. **Gallery construction**: A set of images is processed (excluding the query’s camera), features are extracted, and indexed using FAISS.
7. **Search and comparison**: A k-NN search (k=5) is performed to find the most similar candidates.
   A vehicle is considered correctly identified if the distance to the top result is sufficiently low and the distance variance is significant.
8. **Results visualization**: If the vehicle is identified, the corresponding image is displayed along with the camera ID and vehicle ID.

The system is designed to perform ReID on realistic datasets and can be adapted for execution on either CPU or GPU.
