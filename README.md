# 🚦 Traffic Light Detection — ADAS Consensus System

A real-time traffic light detection system that fuses **YOLOv11 deep learning** with **HSV color analysis** to identify and announce traffic light states. The two methods "discuss" each result before confirming — eliminating false positives and providing reliable audio/visual feedback for driver assistance applications.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [How It Works](#-how-it-works)
4. [Project Structure](#-project-structure)
5. [Software Stack](#-software-stack)
6. [Installation & Setup](#-installation--setup)
7. [Configuration](#-configuration)
8. [Component Details](#-component-details)
9. [UI Overview](#-ui-overview)
10. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

### Purpose
This system is designed as an **Advanced Driver Assistance System (ADAS)** module that detects traffic light states in real time from a video source and responds with:
- Visual bounding boxes with consensus labels on screen
- Status icons indicating the current confirmed signal
- Bilingual audio announcements (Thai 🇹🇭 & English 🇬🇧)

### Key Features
✅ **Dual-Method Verification** — AI and HSV must agree before a result is accepted  
✅ **Real-time Bounding Box Display** — MATCH (green box) or CONFLICT (red box)  
✅ **Black Housing Check** — Filters out non-traffic-light colored objects  
✅ **Audio Announcements** — Thai and English `.wav` voice alerts  
✅ **Status Icon Panel** — Visual red/yellow/green light indicator  
✅ **Cooldown Timer** — Prevents repeated alerts for the same signal  
✅ **Multithreaded Architecture** — Smooth video playback while AI runs in background  
✅ **Simulation Ready** — Works on any `.mov` / `.mp4` video file  

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE                          │
│              (.mov / .mp4 / webcam index)                    │
└──────────────────────────┬───────────────────────────────────┘
                           │  frame
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     MAIN THREAD                              │
│                  run_adas_discussion()                       │
│                                                              │
│   cap.read() → latest_frame ──────────────► display_frame    │
│                    │                              │          │
│                    │ (shared memory)              │          │
└────────────────────┼──────────────────────────────┼──────────┘
                     │                              │
                     ▼                              ▼
┌────────────────────────────────┐  ┌───────────────────────────┐
│       BACKGROUND THREAD        │  │         UI CANVAS         │
│         ai_worker()            │  │       1280 × 720 px       │
│                                │  │                           │
│  1. YOLO Inference (Ai.pt)     │  │  ┌─────────────────────┐  │
│     → bounding box             │  │  │   Video Feed        │  │
│     → class (red/yellow/green) │  │  │   854 × 480 px      │  │
│                                │  │  │   + bbox overlays   │  │
│  2. HSV Color Analysis         │  │  └─────────────────────┘  │
│     → measure R/Y/G pixels     │  │                           │
│     → dominant color           │  │  ┌──────────────────────┐ │
│                                │  │  │   Status Panel       │ │
│  3. Black Housing Check        │  │  │   280 × 280 px icon  │ │
│     → ratio > 15% black        │  │  │   FINAL SIGNAL text  │ │
│                                │  │  └──────────────────────┘ │
│  4. Consensus Logic            │  └───────────────────────────┘
│     YOLO == HSV && has_black?  │
│     → MATCH or CONFLICT        │
└──────────────┬─────────────────┘
               │  latest_result[]
               ▼
┌──────────────────────────────────────────────────────────────┐
│                     OUTPUT ACTIONS                           │
│                                                              │
│   MATCH:    ● Green bbox  ● Update current_status           │
│             ● Play .wav   ● Show status icon                │
│                                                              │
│   CONFLICT: ● Red bbox    ● Show AI:x/HSV:y label           │
│             ● No audio    ● Status unchanged                │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔍 How It Works

### Step 1 — YOLO Detection (AI Perspective)
The custom-trained YOLOv11 model (`Ai/Ai.pt`) runs on every frame captured by the background thread. It detects traffic light regions and predicts a color class:

| Class ID | Color  |
|----------|--------|
| 0        | Green  |
| 1        | Red    |
| 2        | Yellow |

The model uses a confidence threshold of **0.35** — detections below this are discarded.

---

### Step 2 — HSV Color Analysis (Digital Image Processing Perspective)
Once YOLO provides a bounding box, the region of interest (ROI) is extracted and independently analyzed using OpenCV's HSV color space.

**HSV Color Ranges Used**:
```
Red     : Hue [0–10]  + [170–180],  Sat > 120,  Val > 150
Yellow  : Hue [15–35],              Sat > 90,   Val > 150
Green   : Hue [40–90],              Sat > 70,   Val > 100
```

Pixel counts for each color are compared. The dominant color (minimum 20 pixels) is returned as the HSV result.

---

### Step 3 — Black Housing Verification
A dark housing check is applied to the same ROI to confirm it looks like a real traffic light casing:

```
Black mask: Hue [0–180], Sat [0–255], Val [0–100]

If (black pixels / total pixels) > 0.15 → Housing confirmed ✅
```

This step filters out false positives such as colored signboards, brake lights, or neon signs that YOLO might partially detect.

---

### Step 4 — Consensus Logic ("Discussion")

```
┌─────────────────────────────────────────────────────────────┐
│                   CONSENSUS DECISION TABLE                  │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  YOLO says   │  HSV says    │ Black housing │    Result     │
├──────────────┼──────────────┼───────────────┼───────────────┤
│     red      │     red      │      ✅       │ ✅  MATCH     │
│     green    │     green    │      ✅       │ ✅  MATCH     │
│     yellow   │     yellow   │      ✅       │ ✅  MATCH     │
│     red      │     green    │      ✅       │ ❌  CONFLICT  │
│     green    │     red      │      ✅       │ ❌  CONFLICT  │
│     any      │     any      │      ❌       │ ❌  CONFLICT  │
│     any      │   unknown    │    either     │ ❌  CONFLICT  │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

- **MATCH** → Updates `current_status`, plays audio, shows icon
- **CONFLICT** → Red bounding box drawn, status unchanged, no audio

---

### Step 5 — Output: Audio + Visual Feedback

```
[MATCH confirmed]
       │
       ├─► Audio cooldown check (2.5 seconds)
       │      └─ If color has changed since last announcement:
       │             pygame plays thai_<color>.wav
       │
       ├─► Status icon updated
       │      └─ canvas[50:330, 950:1230] = cv2.resize(<color>light.png, (280,280))
       │
       └─► FINAL SIGNAL text printed to canvas
              └─ "FINAL SIGNAL: RED / GREEN / YELLOW"
```

---

## 📁 Project Structure

```
Trafficlight_Detection/
│
├── Detectlight/
│   └── dectectlight.py               # Main application script
│       ├── 🧵 Threads:
│       │   ├── run_adas_discussion()  # Main loop: video capture + UI render
│       │   └── ai_worker()           # Background: YOLO + HSV inference
│       ├── 🔬 Core Functions:
│       │   ├── verify_structure()    # Black housing ratio check
│       │   └── get_hsv_color()       # HSV dominant color analysis
│       └── ⚙️ Config Variables:
│           ├── MAIN_DIR              # Root path (edit this)
│           ├── VIDEO_SPEED_DELAY     # Frame delay in ms (default: 60)
│           └── COOLDOWN              # Audio repeat guard in seconds (default: 2.5)
│
├── Ai/
│   └── Ai.pt                         # Custom YOLOv8 trained weights
│                                     # Classes: 0=green, 1=red, 2=yellow
│
├── image_status/
│   ├── redlight.png                  # Status icon — Red signal
│   ├── yellowlight.png               # Status icon — Yellow signal
│   ├── greenlight.png                # Status icon — Green signal
│   └── Untitled.png                  # Status icon — Unknown / no detection
│
├── sound_status/
│   ├── thai_red.wav                  # 🇹🇭 Thai voice: Red light (DEFAULT)
│   ├── thai_yellow.wav               # 🇹🇭 Thai voice: Yellow light (DEFAULT)
│   ├── thai_green.wav                # 🇹🇭 Thai voice: Green light (DEFAULT)
│   ├── Red light Stop .wav           # 🇬🇧 English alternative
│   ├── Yellow light Please.wav       # 🇬🇧 English alternative
│   ├── The light is green .wav       # 🇬🇧 English alternative
│   └── Green light U can g.wav       # 🇬🇧 English alternative
│
└── foottage/                         # (Create this folder)
    └── testreal.mov                  # Your test video file (place here)
```

---

## 💻 Software Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.0 | YOLOv8 model loading & inference |
| `opencv-python` | ≥ 4.5 | Video capture, HSV analysis, display |
| `pygame` | ≥ 2.0 | `.wav` audio playback |
| `numpy` | ≥ 1.21 | Image array operations |

### Python
- **Minimum**: Python 3.8+
- **Threading**: `threading.Thread` (daemon mode) for background inference
- **I/O**: `cv2.VideoCapture` for video input, `cv2.imshow` for display

### AI Model
- **Framework**: YOLOv8 (Ultralytics)
- **Format**: PyTorch `.pt` weights
- **Input**: BGR video frame
- **Output**: Bounding boxes, class IDs, confidence scores

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- A video file with visible traffic lights (`.mov`, `.mp4`, `.avi`)

### Step 1: Extract the Project

```
Trafficlight_Detection/
  ├── Detectlight/
  ├── Ai/
  ├── image_status/
  ├── sound_status/
  └── foottage/      ← Create this folder manually
```

### Step 2: Install Dependencies

```bash
pip install ultralytics opencv-python pygame numpy
```

### Step 3: Update the Root Path

Open `Detectlight/dectectlight.py` and change `MAIN_DIR` to match your system:

```python
# Line ~10
MAIN_DIR = "/your/path/to/Trafficlight_Detection"
```

**Examples**:
```python
# Windows
MAIN_DIR = "C:/Users/YourName/Documents/Trafficlight_Detection"

# macOS
MAIN_DIR = "/Users/YourName/Desktop/Trafficlight_Detection"

# Linux
MAIN_DIR = "/home/yourname/projects/Trafficlight_Detection"
```

### Step 4: Add Your Test Video

Place a traffic light video at:
```
Trafficlight_Detection/foottage/testreal.mov
```

Or change `VIDEO_PATH` in the script to any supported path:
```python
VIDEO_PATH = "/path/to/your/video.mp4"
```

### Step 5: Run the System

```bash
cd Trafficlight_Detection/Detectlight
python dectectlight.py
```

---

## ⚙️ Configuration

All key settings are near the top of `dectectlight.py`:

```python
# --- Path Configuration ---
MAIN_DIR = "/home/nj/Documents/Project_No.0/Trafficlight_Detection"  # ← Change this

MODEL_PATH  = os.path.join(MAIN_DIR, "Ai/Ai.pt")
VIDEO_PATH  = os.path.join(MAIN_DIR, "foottage/testreal.mov")
IMAGE_DIR   = os.path.join(MAIN_DIR, "image_status")
SOUND_DIR   = os.path.join(MAIN_DIR, "sound_status")
```

### Tunable Parameters

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `VIDEO_SPEED_DELAY` | Global | `60` ms | Delay per frame — lower = faster playback |
| `COOLDOWN` | `run_adas_discussion()` | `2.5` s | Minimum seconds between audio alerts |
| `conf=0.35` | `ai_worker()` | `0.35` | YOLO confidence threshold — raise to reduce false positives |
| `black_ratio > 0.15` | `verify_structure()` | `0.15` | Black housing sensitivity |
| `counts[max_color] > 20` | `get_hsv_color()` | `20` px | Minimum pixels to declare an HSV color |

### Switching to English Audio

The script defaults to Thai voice. To switch to English:

```python
# In the Setup section, change:
sounds = {
    "red":    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Red light Stop .wav")),
    "yellow": pygame.mixer.Sound(os.path.join(SOUND_DIR, "Yellow light Please.wav")),
    "green":  pygame.mixer.Sound(os.path.join(SOUND_DIR, "The light is green .wav")),
}
```

### YOLO Class Mapping

The class IDs must match how `Ai.pt` was trained. Verify and adjust if needed:

```python
# In ai_worker():
ai_names = ["green", "red", "yellow"]   # index 0, 1, 2
```

If detections seem swapped (e.g., red detected as green), reorder this list.

---

## 🔬 Component Details

### `verify_structure(roi)` — Black Housing Check

```python
def verify_structure(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 100])
    mask_black  = cv2.inRange(hsv, lower_black, upper_black)
    black_ratio = cv2.countNonZero(mask_black) / (roi.shape[0] * roi.shape[1])
    return black_ratio > 0.15
```

- **Input**: Cropped bounding box ROI (BGR)
- **Output**: `True` if the region contains ≥15% dark pixels
- **Purpose**: Confirms the detection looks like a physical traffic light housing and not a random colored object

---

### `get_hsv_color(roi)` — HSV Color Classifier

```python
def get_hsv_color(roi):
    # Converts ROI to HSV, applies color masks, returns dominant color
    lower_red1,  upper_red1  = np.array([0,   120, 150]), np.array([10,  255, 255])
    lower_red2,  upper_red2  = np.array([170, 120, 150]), np.array([180, 255, 255])
    lower_yellow,upper_yellow= np.array([15,  90,  150]), np.array([35,  255, 255])
    lower_green, upper_green = np.array([40,  70,  100]), np.array([90,  255, 255])
```

- **Input**: Cropped ROI (BGR)
- **Output**: `"red"`, `"yellow"`, `"green"`, or `"unknown"`
- **Note**: Red uses two HSV ranges because red wraps around the hue wheel (0° and ~180°)

---

### `ai_worker()` — Background Inference Thread

```
[Runs continuously in daemon thread]
       │
       ├─► Read latest_frame (shared global)
       │
       ├─► model.predict(frame, conf=0.35)
       │      └─ Returns boxes, class IDs, confidences
       │
       ├─► For each detected box:
       │      ├─ Extract ROI from frame coordinates
       │      ├─ get_hsv_color(roi)   → hsv_color_guess
       │      ├─ verify_structure(roi) → has_black
       │      └─ is_match = (ai_color == hsv_color) AND has_black
       │
       └─► Write results to latest_result[]
           (consumed by main thread on next frame)
```

---

### `run_adas_discussion(video_source)` — Main Loop

```
[Frame loop]
       │
       ├─► cap.read() → frame
       │
       ├─► latest_frame = frame.copy()  (for ai_worker thread)
       │
       ├─► Read latest_result[] (from ai_worker)
       │
       ├─► Draw bounding boxes:
       │      ├─ MATCH    → Green box, "MATCH: RED/GREEN/YELLOW"
       │      └─ CONFLICT → Red box,   "CONFLICT! AI:x/HSV:y"
       │
       ├─► If MATCH and color changed and cooldown elapsed:
       │      ├─ sounds[color].play()
       │      └─ current_status = color
       │
       ├─► Build canvas (1280×720 white):
       │      ├─ Paste video region  [50:530, 20:874]  (854×480)
       │      └─ Paste status icon   [50:330, 950:1230] (280×280)
       │
       ├─► Render FINAL SIGNAL text
       │
       └─► cv2.imshow() → press Q to quit
```

---

## 🖥️ UI Overview

The display window renders a **1280×720** white canvas with two regions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADAS Consensus Mode (AI + HSV Discussion)            │
├────────────────────────────────────────────┬────────────────────────────┤
│                                            │                            │
│         Live Video Feed                    │    Status Icon             │
│         854 × 480 px                       │    280 × 280 px            │
│                                            │                            │
│  ┌─────── Green box ──────────────────┐    │  [redlight.png     ]  🔴   │
│  │  MATCH: RED                        │    │  [yellowlight.png  ]  🟡   │
│  └────────────────────────────────────┘    │  [greenlight.png   ]  🟢   │
│                                            │  [Untitled.png     ]  ⬜   │
│  ┌─────── Red box ────────────────────┐    ├────────────────────────────┤
│  │  CONFLICT! AI:green/HSV:red        │    │  FINAL SIGNAL: RED         │
│  └────────────────────────────────────┘    │  AI Discussion System Active│
│                                            │                            │
└────────────────────────────────────────────┴────────────────────────────┘
```

### Bounding Box Colors

| Box Color | Label Format | Meaning |
|-----------|-------------|---------|
| 🟩 Green | `MATCH: RED` | AI and HSV agree — result confirmed |
| 🟥 Red | `CONFLICT! AI:green/HSV:red` | Methods disagree — result discarded |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application cleanly |

---

## 🔧 Troubleshooting

### 1. `FileNotFoundError` for model or sound files
```
Error: [Errno 2] No such file or directory: '.../Ai/Ai.pt'
```
**Solution**: Update `MAIN_DIR` in `dectectlight.py` to the correct absolute path on your machine.

---

### 2. Video does not play / `cap.isOpened()` is False
```
Issue: Black screen or immediate exit
```
**Solutions**:
- Confirm the video file exists at `foottage/testreal.mov`
- Verify the filename and extension match exactly (case-sensitive on Linux/macOS)
- Update `VIDEO_PATH` to an absolute path for testing:
  ```python
  VIDEO_PATH = "/absolute/path/to/your/video.mp4"
  ```

---

### 3. No audio output
```
Issue: Detections shown but no sound plays
```
**Solutions**:
- Check system volume and audio output device
- Verify `.wav` files exist in `sound_status/` with exact filenames
- Run `pygame.mixer.init()` check manually:
  ```python
  import pygame
  pygame.mixer.init()
  s = pygame.mixer.Sound("sound_status/thai_red.wav")
  s.play()
  import time; time.sleep(2)
  ```
- If files load but no sound, check OS audio permissions

---

### 4. All detections show `CONFLICT`
```
Issue: Green boxes never appear
```
**Solutions**:
- Verify YOLO class mapping matches your model (`0=green, 1=red, 2=yellow`)
- Lower the black ratio threshold (try `> 0.05` instead of `> 0.15`)
- Reduce confidence threshold: `conf=0.20`
- Print debug info to check what YOLO and HSV actually return:
  ```python
  print(f"AI: {ai_color_guess}, HSV: {hsv_color_guess}, Black: {has_black}")
  ```

---

### 5. High CPU usage / slow performance
```
Issue: Video lag, stuttering, or high fan noise
```
**Solutions**:
- Increase `VIDEO_SPEED_DELAY` (e.g., `90` or `120`)
- Add a longer sleep in `ai_worker`: `time.sleep(0.05)`
- Reduce input resolution before inference:
  ```python
  frame_small = cv2.resize(latest_frame, (640, 360))
  results = model.predict(frame_small, conf=0.35, verbose=False)
  ```
- Run on a machine with a discrete GPU (CUDA-enabled)

---

### 6. `pygame.error: No such file or directory`
```
Error during mixer.Sound() initialization
```
**Solution**: Ensure `SOUND_DIR` resolves correctly. Print it to verify:
```python
print("Sound dir:", SOUND_DIR)
print("Exists:", os.path.exists(SOUND_DIR))
```

---

### 7. `ModuleNotFoundError`
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**:
```bash
pip install ultralytics opencv-python pygame numpy
```
If using a virtual environment, make sure it is activated before running.

---

## 📝 Notes

- The YOLOv11 inference runs in a **daemon background thread** so it does not block video rendering. The main thread always displays the most recent result available.
- The **2.5-second cooldown** applies per color — switching from red to green will play green immediately even if red was just announced.
- The system is designed for **recorded video** but can be adapted for a **webcam** by changing `VIDEO_PATH` to a device index: `cap = cv2.VideoCapture(0)`
- `SIMULATION_MODE` does not exist in this project. To test without a real road video, use any footage containing traffic lights.

---

**Last Updated**: March 2026  
**Project Status**: Active  
**Tested On**: Python 3.10, Ubuntu 22.04 / Windows 11, YOLOv8n
