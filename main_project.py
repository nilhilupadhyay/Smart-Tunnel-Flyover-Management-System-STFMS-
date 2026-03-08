import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os


VIDEO_PATH = 'stock.mp4'
MODEL_NAME = 'yolov8n.pt'
SPEED_LIMIT_KMH = 60 

PIXELS_PER_METER = 20 


STOPPED_TIME_THRESHOLD = 3.0 

STOPPED_PIXEL_THRESHOLD = 5 


if not os.path.exists(VIDEO_PATH):
    print(f"--- DEBUG ERROR ---")
    print(f"Error: Video file not found at this path: {VIDEO_PATH}")
    print("Please make sure the video file is in the same folder as the .py script.")
    exit()
else:
    print(f"--- DEBUG SUCCESS ---")
    print(f"Video file found at: {VIDEO_PATH}")


print("--- DEBUG --- Loading YOLO model...")
model = YOLO(MODEL_NAME)
print("--- DEBUG --- Opening video capture...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)


if not cap.isOpened():
    print(f"--- DEBUG ERROR ---")
    print(f"Error: Could not open video file.")
    print("OpenCV might be missing codecs, or the file is corrupt.")
    exit() 
else:
    print(f"--- DEBUG SUCCESS ---")
    print(f"Video capture is open! Video FPS: {fps}")


track_history = defaultdict(lambda: [])
vehicle_speeds = {}

stopped_vehicles_info = {} 


cv2.namedWindow("Vehicle Monitor", cv2.WINDOW_NORMAL)
print("--- DEBUG --- Created display window.")

print("--- DEBUG --- Starting main processing loop...")
frame_count = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(f"--- DEBUG --- End of video or failed to read frame.")
        break 

    frame_count += 1
    
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

    try:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    except AttributeError:
        
        cv2.imshow("Vehicle Monitor", frame) 
        
       
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        continue

    
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        current_position = (center_x, center_y)

        
        track = track_history[track_id]
        track.append(current_position)
        if len(track) > 1:
            pixel_distance = np.linalg.norm(np.array(track[-1]) - np.array(track[-2]))
            real_world_distance = pixel_distance / PIXELS_PER_METER
            speed_kmh = (real_world_distance * fps) * 3.6
            vehicle_speeds[track_id] = speed_kmh

       
        if track_id in stopped_vehicles_info:
            last_position, stopped_frames = stopped_vehicles_info[track_id]
            #
            if np.linalg.norm(np.array(current_position) - np.array(last_position)) < STOPPED_PIXEL_THRESHOLD:
                stopped_frames += 1 
            else:
                stopped_frames = 0 
            stopped_vehicles_info[track_id] = [current_position, stopped_frames]
        else:
            
            stopped_vehicles_info[track_id] = [current_position, 0]

        
        color = (0, 255, 0) 
        display_text = f"ID: {track_id}"
        
        is_stopped = False
        is_speeding = False

        
        stopped_frames = stopped_vehicles_info[track_id][1]
        stopped_duration = stopped_frames / fps
        if stopped_duration >= STOPPED_TIME_THRESHOLD:
            color = (0, 0, 255) 
            display_text += f" - ALERT: STOPPED ({int(stopped_duration)}s)"
            is_stopped = True

       
        if not is_stopped and track_id in vehicle_speeds:
            current_speed = vehicle_speeds[track_id]
            display_text += f" {int(current_speed)} km/h"
            if current_speed > SPEED_LIMIT_KMH:
                color = (0, 255, 255) # Yellow for speeding
                display_text += " - ALERT: SPEEDING"
                is_speeding = True
        elif not is_stopped:
            display_text += " (calculating...)"

        
       
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            display_text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


    cv2.imshow("Vehicle Monitor", frame)

  
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# --- Cleanup ---
print("--- DEBUG --- Cleaning up and closing windows...")
cap.release()
cv2.destroyAllWindows()
print("--- DEBUG --- Script finished.")
