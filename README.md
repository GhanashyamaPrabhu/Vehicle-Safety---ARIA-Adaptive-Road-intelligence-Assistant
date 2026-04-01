# 🚗 Vehicle Safety AI System
## Jetson Orin Nano + Femto Bolt Camera

Fully offline Vehicle Human-Robot Interaction (HRI) 
safety system running on NVIDIA Jetson Orin Nano.

## Hardware
- NVIDIA Jetson Orin Nano 8GB
- Orbbec Femto Bolt 3D Camera
- NVMe PCIe SSD

## AI Stack
- YOLOv8n (TensorRT FP16) — Object Detection
- Phi-3.5 Mini 4-bit — Scene Description
- Piper TTS — Voice Warnings
- pyorbbecsdk — RGB + Depth Camera

## Features
- 100% Offline — No internet needed
- Real-time object detection at 30fps
- Depth-based distance measurement
- Pedestrian & vehicle warnings
- AI driving instructions
- Live HUD display + Depth map

## Safety Zones
- SAFE    → obstacle > 5m
- CAUTION → obstacle 3-5m  
- WARNING → obstacle 1.5-3m
- DANGER  → obstacle < 1.5m

## Setup
```bash
# Clone repo
git clone https://github.com/GhanashyamaPrabhu/vehicle-safety-ai.git

# Install dependencies
pip install -r requirements.txt

# Run
python3 main.py
```

## System Status
| Component | Version |
|---|---|
| JetPack | 6.2.2 |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| PyTorch | 2.10.0 |
| Ubuntu | 22.04.5 LTS |
