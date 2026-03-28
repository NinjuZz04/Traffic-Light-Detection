# 🚦 Traffic Light Detection — ADAS Consensus System

A real-time traffic light detection system that combines **YOLOv8 AI object detection** with **HSV color analysis** to verify and announce traffic light states. The two methods cross-check each other ("discuss") before confirming a result, reducing false detections.

---

## How It Works

The system uses a dual-verification approach called **AI Discussion Mode**:

### 1. AI Detection (YOLO)
A custom-trained YOLOv8 model (`Ai/Ai.pt`) scans each video frame to locate traffic lights and predict their color class:
- Class 0 → Green
- Class 1 → Red
- Class 2 → Yellow

### 2. HSV Color Verification (Digital Image Processing)
Once YOLO finds a bounding box, the cropped region (ROI) is analyzed using OpenCV's HSV color space. It independently determines the lit color by measuring pixel coverage for red, yellow, and green hue ranges.

Additionally, a **black pixel check** verifies the ROI contains the dark housing typical of a real traffic light, filtering out false positives from colored signs or lights.

### 3. Consensus Logic
Both results are compared:

| YOLO says | HSV says | Black housing | Result |
|-----------|----------|---------------|--------|
| red | red | ✅ | ✅ **MATCH** — confirmed red |
| green | green | ✅ | ✅ **MATCH** — confirmed green |
| red | green | ✅ | ❌ **CONFLICT** — ignored |
| any | any | ❌ | ❌ **CONFLICT** — ignored |

Only a **MATCH** triggers a status update and plays the audio announcement. A **CONFLICT** is displayed on screen but does not change the current status, preventing false alerts.

### 4. Audio & Visual Feedback
- On confirmed detection, a `.wav` voice announcement plays (Thai and English voices available).
- A status icon (red/yellow/green light image) is shown in the UI panel.
- A **2.5-second cooldown** prevents repeated announcements for the same light.

---

## Project Structure

```
Trafficlight_Detection/
│
├── Detectlight/
│   └── dectectlight.py        # Main application script
│
├── Ai/
│   └── Ai.pt                  # Custom YOLOv8 trained model
│
├── image_status/
│   ├── redlight.png
│   ├── yellowlight.png
│   ├── greenlight.png
│   └── Untitled.png           # Unknown/no detection icon
│
├── sound_status/
│   ├── thai_red.wav
│   ├── thai_yellow.wav
│   ├── thai_green.wav
│   ├── Red light Stop .wav
│   ├── Yellow light Please.wav
│   ├── The light is green .wav
│   └── Green light U can g.wav
│
└── foottage/
    └── testreal.mov           # Test video (add your own here)
```

---

## Requirements

Install dependencies with pip:

```bash
pip install ultralytics opencv-python pygame numpy
```

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 model inference |
| `opencv-python` | Video capture, HSV analysis, display |
| `pygame` | Audio playback |
| `numpy` | Image array operations |

> Python 3.8 or higher is recommended.

---

## Setup & Usage

### 1. Clone / Extract the project
Extract the zip so the folder structure matches the tree above.

### 2. Update the path in the script
Open `Detectlight/dectectlight.py` and update `MAIN_DIR` to point to your own folder:

```python
# Line ~10 in dectectlight.py
MAIN_DIR = "/your/path/to/Trafficlight_Detection"
```

### 3. Add a test video
Place your video file at:
```
Trafficlight_Detection/foottage/testreal.mov
```
Or update `VIDEO_PATH` in the script to point to any video you want to use.

### 4. Run the script

```bash
cd Trafficlight_Detection/Detectlight
python dectectlight.py
```

### 5. Controls
| Key | Action |
|-----|--------|
| `Q` | Quit the application |

---

## UI Overview

The display window is a **1280×720 canvas** with two panels:

- **Left panel** — Live video feed with bounding boxes drawn around detected traffic lights.
  - 🟩 Green box + `MATCH: RED/GREEN/YELLOW` → consensus reached
  - 🟥 Red box + `CONFLICT! AI:x/HSV:y` → disagreement, result ignored

- **Right panel** — Status icon showing the current confirmed traffic light color, plus a text label (`FINAL SIGNAL: RED / GREEN / YELLOW`).

---

## Configuration

You can tune the following variables at the top of `dectectlight.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SPEED_DELAY` | `60` ms | Delay between frames (lower = faster playback) |
| `COOLDOWN` | `2.5` s | Minimum time between audio announcements |
| `conf=0.35` | 0.35 | YOLO confidence threshold (raise to reduce false positives) |

---

## Notes

- The system runs YOLO inference on a **background thread** to keep the video display smooth.
- Sound files include both **Thai** (`thai_*.wav`) and **English** voice announcements — the script defaults to Thai. To switch to English, update the `sounds` dictionary in the script to reference the English `.wav` files.
- The YOLO class mapping (`0:green, 1:red, 2:yellow`) must match how your `Ai.pt` model was trained. Verify this if detections seem misclassified.
