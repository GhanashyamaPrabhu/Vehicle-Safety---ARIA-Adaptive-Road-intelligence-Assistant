# 🚗 Vehicle Safety AI System
### Offline Edge AI on NVIDIA Jetson Orin Nano + Orbbec Femto Bolt

![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-green)
![JetPack](https://img.shields.io/badge/JetPack-6.2.2-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-12.6-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Offline](https://img.shields.io/badge/Mode-100%25%20Offline-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

A fully **offline**, **GPU-accelerated** Vehicle Human-Robot 
Interaction (HRI) Safety System that runs entirely on the 
NVIDIA Jetson Orin Nano 8GB edge device. No internet 
connection is required after initial setup.

The system uses an Orbbec Femto Bolt 3D camera mounted on 
a vehicle to:
- Detect pedestrians, vehicles and obstacles in real-time
- Measure exact distances using depth sensing
- Generate AI-powered driving instructions
- Display a live HUD with safety alerts
- Provide voice warnings via TTS (when speaker connected)

---

## 🎯 Use Case

This system is designed to be mounted on a vehicle and acts 
as an intelligent co-pilot that:

- Warns the driver about nearby pedestrians
- Detects vehicles and obstacles ahead
- Checks if the road ahead is drivable
- Gives real-time AI instructions like a GPS voice
- Alerts pedestrians nearby that a vehicle is approaching

---

## 🖥️ Hardware

| Component | Details |
|---|---|
| **Edge Computer** | NVIDIA Jetson Orin Nano 8GB |
| **Camera** | Orbbec Femto Bolt 3D (RGB + Depth) |
| **Storage** | NVMe PCIe Gen3 SSD |
| **OS** | Ubuntu 22.04.5 LTS |
| **JetPack** | 6.2.2 (flashed via SDK Manager) |
| **Power Mode** | 15W MAXN (maximum performance) |
| **RAM** | 8GB + 8GB swap |

---

## 🧠 AI Pipeline
```
Orbbec Femto Bolt Camera
        │
        ├── RGB Stream  (1920x1080 @ 30fps)
        └── Depth Stream (640x576  @ 30fps)
                │
                ▼
        ┌───────────────────┐
        │  YOLOv8n          │  ← Object Detection
        │  TensorRT FP16    │  ← GPU Accelerated
        │  8.8MB engine     │  ← Real-time 15fps
        └───────────────────┘
                │
                ▼
        ┌───────────────────┐
        │  Depth Analysis   │  ← Distance per object
        │  Safety Zones     │  ← SAFE/CAUTION/WARNING/DANGER
        │  Road Check       │  ← Is road drivable?
        └───────────────────┘
                │
                ▼
        ┌───────────────────┐
        │  Phi-3.5 Mini     │  ← Scene Understanding
        │  4-bit NF4        │  ← Quantized LLM
        │  3.8B parameters  │  ← ~3.5GB GPU RAM
        └───────────────────┘
                │
                ▼
        ┌───────────────────┐
        │  Piper TTS        │  ← Voice Warnings
        │  en_US-lessac     │  ← CPU inference
        │  Offline voice    │  ← No cloud needed
        └───────────────────┘
                │
                ▼
        ┌───────────────────┐
        │  HUD Display      │  ← Live overlay
        │  Depth Map        │  ← Color visualization
        │  Alert System     │  ← Visual + Voice
        └───────────────────┘
```

---

## 🤖 AI Models Used

| Model | Purpose | Size | Format | GPU |
|---|---|---|---|---|
| **YOLOv8n** | Object detection | 8.8MB | TensorRT FP16 | ✅ |
| **Phi-3.5 Mini** | Scene description | 7.64GB | 4-bit NF4 | ✅ |
| **Piper TTS** | Voice output | ~60MB | ONNX | CPU |

### YOLOv8n — Object Detection
- Detects 80 object classes
- Runs as TensorRT FP16 engine
- Optimized specifically for Jetson Orin GPU
- Detects: people, vehicles, obstacles, animals
- Confidence threshold: 35%

### Phi-3.5 Mini Instruct (Microsoft)
- 3.8 billion parameter language model
- Loaded in 4-bit NF4 quantization
- Uses bitsandbytes Jetson build
- Generates driving instructions
- Runs fully on Jetson GPU (CUDA)
- attn_implementation: eager (no flash attention)

### Piper TTS
- Offline text-to-speech engine
- Voice: en_US-lessac-medium
- Runs on CPU
- No internet needed
- Requires speaker/audio output

---

## 🚦 Safety Zone System
```
Distance    Zone       Action
─────────────────────────────────────────
> 5.0m   →  SAFE      ✅ Proceed normally
3.0-5.0m →  CAUTION   ⚠️  Reduce speed
1.5-3.0m →  WARNING   ⚠️  Slow down now
< 1.5m   →  DANGER    🛑 Stop immediately
```

### Object Priority
```
🔴 RED    → Pedestrians (person, bicycle, motorcycle)
🟠 ORANGE → Vehicles    (car, truck, bus, train)
🟡 YELLOW → Obstacles   (signs, animals, furniture)
```

---

## 🖥️ Live Display

### Window 1 — Vehicle Safety HUD
```
┌─────────────────────────────────────────────┐
│ [WARNING]  Nearest: 2.1m   Road: 3.5m  [30]│
│                                      km/h   │
│  ┌──────────────┐  Persons:  2             │
│  │ person 2.1m  │  Vehicles: 1             │
│  │   89%        │  Objects:  0             │
│  └──────────────┘  13:09:42               │
│                                             │
│  ┌──────────────┐                          │
│  │ car 4.2m 92% │                          │
│  └──────────────┘                          │
│                                             │
│ AI: Slow down, pedestrian ahead at 2.1m.   │
└─────────────────────────────────────────────┘
```

### Window 2 — Depth Map
```
🔵 Blue  = Far away  (safe)
🟢 Green = Medium    (caution)
🔴 Red   = Close     (danger)
```

---

## 📦 Software Stack

| Package | Version | Purpose |
|---|---|---|
| JetPack | 6.2.2 | NVIDIA platform SDK |
| Ubuntu | 22.04.5 LTS | Operating System |
| CUDA | 12.6 | GPU computing |
| TensorRT | 10.3.0 | Model optimization |
| cuDNN | 9.3.0.75 | Deep learning |
| PyTorch | 2.10.0+cu126 | AI framework |
| Transformers | 4.44.0 | LLM loading |
| bitsandbytes | 0.48.0 (Jetson) | 4-bit quantization |
| Ultralytics | 8.4.33 | YOLO framework |
| OpenCV | 4.13.0 | Image processing |
| ONNX Runtime | 1.23.0 | Model inference |
| pyorbbecsdk | 2.0.18 | Camera SDK |
| Piper TTS | latest | Voice synthesis |
| numpy | 2.2.6 | Array processing |
| scipy | 1.15.3 | Scientific computing |

---

## ⚡ Performance

| Component | Performance |
|---|---|
| Camera RGB | 1920x1080 @ 30fps |
| Camera Depth | 640x576 @ 30fps |
| YOLO Detection | ~15 FPS on GPU |
| Depth Processing | Real-time |
| LLM Inference | ~15-20 sec/response |
| TTS Generation | ~2 sec (when speaker connected) |
| Total GPU RAM | ~4.3GB / 8GB |
| Swap Used | ~26MB / 8GB |

---

## 📊 GPU Memory Budget
```
Jetson Orin Nano 8GB Total RAM
├── Ubuntu OS + Desktop:    ~2.0GB
├── YOLOv8n TRT FP16:       ~0.8GB  ✅ GPU
├── Phi-3.5 Mini 4-bit:     ~3.5GB  ✅ GPU
├── Camera buffers:          ~0.2GB
├── System overhead:         ~0.5GB
└── Available headroom:      ~1.0GB ✅ Safe
```

---

## 🛠️ Complete Setup Journey

### Step 1 — Flash JetPack via SDK Manager
```bash
# On host PC (Ubuntu 22.04 x86)
# Download SDK Manager 2.4.0 from NVIDIA
sudo apt install ./sdkmanager_2.4.0-*_amd64.deb

# Put Jetson in Force Recovery Mode
# Short FC_REC + GND pins while powering on
# Select: JetPack 6.2.2, Jetson Orin Nano 8GB
# Storage: NVMe
# Flash and install all components
```

### Step 2 — System Optimization
```bash
# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Add 8GB swap for LLM
sudo fallocate -l 8G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# Install system dependencies
sudo apt update && sudo apt install -y \
  python3-pip cmake ninja-build \
  libusb-1.0-0-dev libudev-dev \
  python3-venv git wget curl \
  libopenblas-dev python3-dev \
  python3.10-venv build-essential
```

### Step 3 — Python Environment
```bash
# Create virtual environment
python3 -m venv ~/vla_env
source ~/vla_env/bin/activate
echo 'source ~/vla_env/bin/activate' >> ~/.bashrc

# Install PyTorch (Jetson optimized)
pip install torch torchvision \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Fix libcudss dependency
pip install nvidia-cudss-cu12
echo 'export LD_LIBRARY_PATH=~/vla_env/lib/python3.10/\
site-packages/nvidia/cu12/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Install AI packages
pip install transformers==4.44.0 accelerate
pip install bitsandbytes \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install ultralytics
pip install onnxruntime-gpu \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install opencv-python numpy scipy
pip install piper-tts pathvalidate
pip install huggingface_hub
```

### Step 4 — Link System TensorRT to venv
```bash
# TensorRT is installed by JetPack at system level
# Copy to venv so Python can find it
cp -r /usr/lib/python3.10/dist-packages/tensorrt \
  ~/vla_env/lib/python3.10/site-packages/
cp -r /usr/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info \
  ~/vla_env/lib/python3.10/site-packages/

# Verify
python3 -c "import tensorrt as trt; print(trt.__version__)"
# Output: 10.3.0 ✅
```

### Step 5 — Download AI Models
```bash
mkdir -p ~/models/{phi35,piper,yolo}

# YOLOv8n
cd ~/models/yolo
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Phi-3.5 Mini (~7.64GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/Phi-3.5-mini-instruct',
    local_dir='/home/yahboom/models/phi35',
    ignore_patterns=['*.gguf','*.ggml']
)"

# Piper TTS voice
cd ~/models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/\
main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/\
main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### Step 6 — Export YOLOv8 to TensorRT
```bash
cd ~/models/yolo
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(
    format='engine',
    half=True,
    device=0,
    imgsz=640,
    workspace=2
)"
# Takes ~7.5 minutes
# Creates yolov8n.engine (8.8MB) ✅
```

### Step 7 — Build Femto Bolt Camera SDK
```bash
# Clone pyorbbecsdk
cd ~
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk

# Install build deps
pip install pybind11[global]
sudo apt install cmake build-essential \
  libusb-1.0-0-dev libudev-dev -y

# Build with CMake
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c \
  "import pybind11; print(pybind11.get_cmake_dir())") ..
make -j4
make install

# Install Python package
cd ~/pyorbbecsdk
pip install -e .

# Copy .so to venv
cp install/lib/pyorbbecsdk.cpython-310-aarch64-linux-gnu.so \
  ~/vla_env/lib/python3.10/site-packages/
cp install/lib/libOrbbecSDK.so \
  ~/vla_env/lib/python3.10/site-packages/
cp -r install/lib/extensions \
  ~/vla_env/lib/python3.10/site-packages/

# USB permissions for camera
sudo cp scripts/env_setup/99-obsensor-libusb.rules \
  /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG plugdev yahboom
sudo usermod -aG video yahboom

# Add library path
echo 'export LD_LIBRARY_PATH=/home/yahboom/\
pyorbbecsdk/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=/home/yahboom/pyorbbecsdk:\
/home/yahboom/pyorbbecsdk/install/lib:$PYTHONPATH' \
>> ~/.bashrc
```

### Step 8 — Run the System
```bash
source ~/vla_env/bin/activate
cd ~/hri_vla
python3 main.py
```

---

## 📁 Project Structure
```
~/hri_vla/
├── main.py              ← Main pipeline
├── README.md            ← This file
├── requirements.txt     ← Python packages
└── .gitignore           ← Git ignore rules

~/models/
├── yolo/
│   ├── yolov8n.pt       ← Original PyTorch model
│   └── yolov8n.engine   ← TensorRT FP16 engine ✅
├── phi35/               ← Phi-3.5 Mini (7.64GB)
│   ├── config.json
│   ├── tokenizer.json
│   └── model-*.safetensors
└── piper/
    ├── en_US-lessac-medium.onnx
    └── en_US-lessac-medium.onnx.json

~/pyorbbecsdk/           ← Camera SDK (built from source)
~/vla_env/               ← Python virtual environment
```

---

## 🔊 Adding Speaker (Future)
```bash
# Bluetooth speaker
bluetoothctl power on
bluetoothctl scan on
bluetoothctl pair XX:XX:XX:XX:XX:XX
bluetoothctl connect XX:XX:XX:XX:XX:XX

# Then in main.py change:
SPEAKER_CONNECTED = True
```

---

## 🚀 Future Improvements

- [ ] Add GPS speed → real speed limit enforcement
- [ ] Add night/IR mode → low light detection  
- [ ] Add event recording → save dangerous moments
- [ ] Add ROS2 → connect to robot/vehicle actuators
- [ ] Add web dashboard → remote monitoring
- [ ] Add lane detection → road boundary awareness
- [ ] Add traffic sign recognition → sign-specific warnings
- [ ] Add multi-camera → 360° coverage

---

## ⚠️ Known Issues & Fixes

| Issue | Fix |
|---|---|
| `nvcc not found` | `echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc` |
| `libcudss.so not found` | `pip install nvidia-cudss-cu12` |
| `tensorrt not found` | Copy from `/usr/lib/python3.10/dist-packages/` |
| `pyorbbecsdk no Context` | Must activate venv first |
| `Camera permission denied` | Run udev rules + add to plugdev group |
| `4-bit model .to() error` | Use `device_map='auto'` not `'cuda'` |
| `flash-attention warning` | Harmless, using `attn_implementation='eager'` |

---

## 📜 License
MIT License — Free to use and modify

## 👤 Author
**GhanashyamaPrabhu**
GitHub: https://github.com/GhanashyamaPrabhu

## 👥 Contributors
- **Bhavya Oza** — [@bpo7912](https://github.com/bpo7912)
- **Devam Shah** — [@Devamshah03](https://github.com/Devamshah03)

## 🙏 Acknowledgements
- NVIDIA Jetson Team — JetPack & TensorRT
- Microsoft — Phi-3.5 Mini model
- Ultralytics — YOLOv8
- Orbbec — Femto Bolt SDK
- Jetson AI Lab — Optimized packages
- Rhasspy — Piper TTS
