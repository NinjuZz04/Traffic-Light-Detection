import cv2
import numpy as np
import pygame
import os
import time
import threading
from ultralytics import YOLO

# --- Path Configuration ---
# Defining all main directories explicitly so no NameErrors occur
MAIN_DIR = "/home/nj/Documents/Project_No.0/Trafficlight_Detection"
BASE_DIR = os.path.join(MAIN_DIR, "Detectlight")
IMAGE_DIR = os.path.join(MAIN_DIR, "image_status")
SOUND_DIR = os.path.join(MAIN_DIR, "sound_status")

# Pointing to the Ai and foottage folders inside the main directory
MODEL_PATH = os.path.join(MAIN_DIR, "Ai/Ai.pt")
VIDEO_PATH = os.path.join(MAIN_DIR, "your_foottage_trafficlighr")

# --- Setup ---
pygame.mixer.init()
sounds = {
    "red": pygame.mixer.Sound(os.path.join(SOUND_DIR, "thai_red.wav")), 
    "yellow": pygame.mixer.Sound(os.path.join(SOUND_DIR, "thai_yellow.wav")), 
    "green": pygame.mixer.Sound(os.path.join(SOUND_DIR, "thai_green.wav"))
}

status_images = {"red": "redlight.png", "yellow": "yellowlight.png", "green": "greenlight.png", "unknown": "Untitled.png"}
model = YOLO(MODEL_PATH)

# --- Global Variables ---
latest_frame = None
latest_result = None
is_running = True
inference_time = 0 
VIDEO_SPEED_DELAY = 60 

def verify_structure(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100]) 
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    black_ratio = cv2.countNonZero(mask_black) / (roi.shape[0] * roi.shape[1])
    return black_ratio > 0.15

def get_hsv_color(roi):
    """ Pure Digital Image Processing Result """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 120, 150]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 150]), np.array([180, 255, 255])
    lower_yellow, upper_yellow = np.array([15, 90, 150]), np.array([35, 255, 255])
    lower_green, upper_green = np.array([40, 70, 100]), np.array([90, 255, 255])

    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    counts = {"red": cv2.countNonZero(mask_r), "yellow": cv2.countNonZero(mask_y), "green": cv2.countNonZero(mask_g)}
    max_color = max(counts, key=counts.get)
    return max_color if counts[max_color] > 20 else "unknown"

def ai_worker():
    global latest_frame, latest_result, is_running, inference_time
    while is_running:
        if latest_frame is not None:
            start_time = time.time()
            # YOLO acts as the AI perspective
            results = model.predict(latest_frame, conf=0.35, verbose=False) 
            inference_time = (time.time() - start_time) * 1000 
            
            temp_results = []
            if len(results) > 0:
                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    # Map YOLO class ID to color names (Assumes 0:green, 1:red, 2:yellow - verify your best.pt)
                    ai_names = ["green", "red", "yellow"] 
                    ai_color_guess = ai_names[cls_id] if cls_id < len(ai_names) else "unknown"
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    roi = latest_frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Digital Image Processing perspective
                        hsv_color_guess = get_hsv_color(roi)
                        has_black = verify_structure(roi)
                        
                        # Discussion Logic: Compare AI vs HSV
                        is_match = (ai_color_guess == hsv_color_guess) and has_black
                        
                        temp_results.append({
                            "box": (x1, y1, x2, y2),
                            "ai_color": ai_color_guess,
                            "hsv_color": hsv_color_guess,
                            "verified": is_match,
                            "conf": conf
                        })
            latest_result = temp_results
        time.sleep(0.01)

def run_adas_discussion(video_source):
    global latest_frame, latest_result, is_running, inference_time
    cap = cv2.VideoCapture(video_source)
    last_detect_time = 0
    COOLDOWN = 2.5 
    current_status = "unknown"

    threading.Thread(target=ai_worker, daemon=True).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        latest_frame = frame.copy()
        now = time.time()
        display_frame = frame.copy()

        if latest_result:
            for res in latest_result:
                x1, y1, x2, y2 = res["box"]
                ai_c = res["ai_color"]
                hsv_c = res["hsv_color"]
                is_ok = res["verified"]
                
                # Visual Discussion Output
                if is_ok:
                    color_box = (0, 255, 0) # Green for Consensus
                    label = f"MATCH: {ai_c.upper()}"
                    
                    if ai_c != current_status and (now - last_detect_time > COOLDOWN):
                        sounds[ai_c].play() 
                        current_status = ai_c
                        last_detect_time = now
                else:
                    color_box = (0, 0, 255) # Red for Logic Error
                    label = f"CONFLICT! AI:{ai_c}/HSV:{hsv_c}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display_frame, label, (x1, y1-10), 0, 0.5, color_box, 2)

        # UI Drawing
        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        canvas[50:530, 20:874] = cv2.resize(display_frame, (854, 480))
        
        icon_name = status_images.get(current_status, "Untitled.png")
        icon_path = os.path.join(IMAGE_DIR, icon_name) 
        
        if os.path.exists(icon_path):
            canvas[50:330, 950:1230] = cv2.resize(cv2.imread(icon_path), (280, 280))

        cv2.putText(canvas, f"FINAL SIGNAL: {current_status.upper()}", (950, 400), 0, 0.7, (0,0,0), 2)
        cv2.putText(canvas, "AI Discussion System Active", (920, 430), 0, 0.5, (50,50,50), 1)

        cv2.imshow("ADAS Consensus Mode (AI + HSV Discussion)", canvas)
        if cv2.waitKey(VIDEO_SPEED_DELAY) & 0xFF == ord('q'): 
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_adas_discussion(VIDEO_PATH)
