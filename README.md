# Neural Pilot Cloning: End-to-End Autonomous Driving

An end-to-end deep learning system for autonomous steering in a driving simulator, inspired by **Nvidia’s PilotNet** architecture. The project implements a full pipeline from data acquisition and behavioral cloning to real-time inference in a closed-loop simulation.

## Project Overview

The goal is to map raw front-facing camera pixels directly to steering commands (**end-to-end learning**) without handcrafted features or lane-detection heuristics. The model is trained by cloning human driving behavior and is regularized through targeted data augmentation so it generalizes to unseen track segments.

### Key Technical Components

- **Behavioral cloning:** Supervised regression from images to steering angle using human driving logs (`driving_log.csv`). 
- **Augmentation for recovery:** Geometric shifts, brightness changes, and flips with label adjustment to synthesize recovery maneuvers from off-center positions. 
- **Focused preprocessing:** Cropping to the road ROI, resizing to \(66 \times 200\), and RGB→YUV conversion to match the original PilotNet input specification.
- **Low-latency inference:** Deployed via a Flask + Socket.IO server that streams frames, applies the same preprocessing, and returns steering predictions to the simulator loop. 
---

## System Architecture

The pipeline is organized into three stages: **data ingestion**, **model training**, and **autonomous inference**. 

### 1. Data distribution

The dataset contains thousands of synchronized camera frames and steering angles. To reduce the natural bias toward straight driving, the steering distribution is inspected and rebalanced via augmentation. 

<img width="600" height="400" alt="steering_hist" src="https://github.com/user-attachments/assets/27b12155-6313-4e4c-9099-f26747d7c8dc" />

> **Figure 1:** Steering angle histogram after augmentation. The distribution is flattened around zero to avoid a trivial “drive straight” solution. 

### 2. Model architecture (PilotNet-based)

The network adapts PilotNet to the simulator resolution and acts as a feature extractor followed by a compact control head. 

| Layer | Output shape | Params | Role |
| :--- | :--- | :--- | :--- |
| Input | (66, 200, 3) | 0 | YUV normalized frame |
| Conv2D | (31, 98, 24) | 1,824 | Strided feature extraction |
| Conv2D | (14, 47, 36) | 21,636 | Strided feature extraction |
| Conv2D | (5, 22, 48) | 43,248 | Strided feature extraction |
| Conv2D | (3, 20, 64) | 27,712 | Local refinement |
| Conv2D | (1, 18, 64) | 36,928 | Local refinement |
| Flatten | (1152) | 0 | Latent representation |
| Dense | (100) | 115,300 | Control layer |
| Dense | (50) | 5,050 | Control layer |
| Dense | (10) | 510 | Control layer |
| Output | (1) | 11 | Steering angle regression |

**Total parameters:** ~250k.

### 3. Image preprocessing

Raw simulator frames include sky, trees, and hood elements that do not contribute to steering. The preprocessing pipeline isolates the road surface before inference. 

1. Crop upper and lower regions to keep only the road ROI. 
2. Resize to \(200 \times 66\) for efficient training and inference.  
3. Convert RGB to YUV to match the color space used in PilotNet.

![right_2025_12_10_22_37_20_716](https://github.com/user-attachments/assets/3bc991ff-8e86-40cb-ab9c-83c2cb64f0cb)

---

## Installation and usage

### Prerequisites

- Python 3.7+  
- Udacity self-driving car simulator (behavioral cloning project).

### Setup

```bash
git clone https://github.com/NaveedAhmeds/Neural-Pilot-Cloning.git

cd Neural-Pilot-Cloning

pip install -r requirements.txt
```

## Running the project

### 1. Model training

To train a steering-angle model from recorded simulator data:

```bash
python src/train.py
```

The training script expects the Udacity-style behavioral cloning format: ```data/driving_log.csv``` plus the corresponding image directory under ```data/IMG/```. It applies cropping, resizing, YUV conversion, and on‑the‑fly augmentation before optimizing the PilotNet‑style CNN with an MSE regression objective. The resulting weights are stored as ```models/model.h5```.

### 2. Autonomous driving (inference server)

To deploy the trained model in closed loop with the simulator:

```bash
python src/drive.py models/model.h5
```

This launches a ```Flask/Socket.IO``` service that accepts frames streamed from the simulator, runs them through the same preprocessing pipeline, performs a forward pass through the network, and returns the predicted steering angle to the simulation client on each timestep. Start the Udacity simulator, select Autonomous Mode, and connect to the server using the host/port configured in drive.py.

