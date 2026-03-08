import cv2
import os
import time
import datetime
import math
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, Response, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO


try:
    from anpr import initialize_anpr, get_plate_from_frame
except ImportError:
    print("[WARNING] 'anpr.py' not found. Using dummy ANPR functions.")
    def initialize_anpr(): pass
    def get_plate_from_frame(frame, coords): return "UNKNOWN"


VIDEO_PATH1 = "stock.mp4" 
VIDEO_PATH2 = "british_highway_traffic.mp4"
VIDEO_PATH3 = "15 minutes of heavy traffic noise in India _ 14-08-2022.mp4"

MODEL_NAME = "yolov8n.pt"

CAMERAS = [
    {
        "id": 0, "name": "Tunnel Entrance", "source": VIDEO_PATH1
    },
    {
        "id": 1, "name": "Flyover Exit",    "source": VIDEO_PATH2
    },
    {
        "id": 2, "name": "In traffic",      "source": VIDEO_PATH3
    }
]

STOPPED_TIME_THRESHOLD = 10.0 



MOVEMENT_RATIO_THRESHOLD = 0.12


FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS = 0.7, 2, 2
CLASS_NAMES = {2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///alerts.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class AlertLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    camera_name = db.Column(db.String(50))
    event_type = db.Column(db.String(50))
    vehicle_id = db.Column(db.Integer)
    details = db.Column(db.String(200))

def log_alert(cam_name, event_type, vid, details):
    with app.app_context():
        try:
            db.session.add(AlertLog(camera_name=cam_name, event_type=event_type, vehicle_id=vid, details=details))
            db.session.commit()
        except Exception as e:
            print("[DB] Error:", e)
            db.session.rollback()

initialize_anpr()
live_alerts = [] 
camera_states = {} 

def get_camera_state(cam_id):
    if cam_id not in camera_states:
        print(f"[INFO] Initializing YOLO model for Camera {cam_id}...")
        try:
            local_model = YOLO(MODEL_NAME)
        except Exception as e:
            print(f"[ERROR] Could not load YOLO model for Cam {cam_id}: {e}")
            local_model = None

        camera_states[cam_id] = {
            "model": local_model, 
            # Stores list of (timestamp, centroid_x, centroid_y, box_diagonal_size)
            "track_history": defaultdict(list), 
            "active_alerts": set(),
            "plate_history": {}
        }
    return camera_states[cam_id]


def generate_frames(cam_id):
    cam_config = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if not cam_config: return

    source = cam_config["source"]
    if isinstance(source, str) and not os.path.exists(source):
        print(f"[WARN] Video file {source} not found. Using Webcam 0.")
        source = 0

    cap = cv2.VideoCapture(source)
    
    state = get_camera_state(cam_id)
    model = state["model"]
    track_history = state["track_history"]
    active_alerts = state["active_alerts"]
    plate_history = state["plate_history"]

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            track_history.clear(); active_alerts.clear(); plate_history.clear()
            continue

        curr_time = time.time()

        if model:
            results = model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)
            
            boxes, ids, clss = [], [], []
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()

            current_ids = set(ids)
            
            for (x1,y1,x2,y2), tid, cid in zip(boxes, ids, clss):
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                
                # Calculate Centroid
                cx, cy = (x1+x2)//2, (y1+y2)//2
                
                # Calculate Diagonal Size (Proxy for vehicle size on screen)
                w, h = x2-x1, y2-y1
                diagonal_size = math.sqrt(w**2 + h**2)

                # --- TRACKING LOGIC (SLIDING WINDOW) ---
                history = track_history[tid]
                history.append((curr_time, cx, cy, diagonal_size))

                # Keep only recent history (last 6 seconds)
                history[:] = [h for h in history if (curr_time - h[0]) <= (STOPPED_TIME_THRESHOLD + 1.0)]

                is_stopped = False
                status_color = (0, 255, 0) # Green (Moving)
                
                # We need data from at least 'STOPPED_TIME_THRESHOLD' seconds ago to compare
                if len(history) > 1:
                    oldest = history[0] # (t, x, y, size)
                    oldest_time, oldest_x, oldest_y, oldest_size = oldest
                    
                    time_diff = curr_time - oldest_time
                    
                    if time_diff >= STOPPED_TIME_THRESHOLD:
                        # Distance moved in screen pixels
                        dist_moved = math.sqrt((cx - oldest_x)**2 + (cy - oldest_y)**2)
                        
                        # Normalize: How much did it move relative to its size?
                        # Using the average size between then and now
                        avg_size = (diagonal_size + oldest_size) / 2.0
                        if avg_size > 0:
                            move_ratio = dist_moved / avg_size
                        else:
                            move_ratio = 1.0 # Avoid div zero

                        # THE CORE CHECK: relative movement
                        if move_ratio < MOVEMENT_RATIO_THRESHOLD:
                            is_stopped = True
                            status_color = (0, 0, 255) # Red (Stopped)

                # --- ALERT LOGIC ---
                cls_name = CLASS_NAMES.get(cid, "Veh")
                label = f"{cls_name} {tid}"

                if is_stopped:
                    label += " STOP"
                    
                    if f"STOP-{tid}" not in active_alerts:
                        # Capture Plate
                        if tid not in plate_history:
                            plate_history[tid] = get_plate_from_frame(frame, (x1,y1,x2,y2))
                        
                        msg = f"{cls_name} ID:{tid} STOPPED. Plate: {plate_history[tid]}"
                        
                        # Insert into global alert list
                        live_alerts.insert(0, {
                            "type":"stop", 
                            "camera":cam_config["name"], 
                            "time":datetime.datetime.now().strftime("%H:%M:%S"), 
                            "message":msg, 
                            "tts": f"Alert. {cls_name} stopped at {cam_config['name']}."
                        })
                        log_alert(cam_config["name"], "STOPPED", tid, msg)
                        active_alerts.add(f"STOP-{tid}")

                cv2.rectangle(frame, (x1,y1), (x2,y2), status_color, BOX_THICKNESS)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, status_color, FONT_THICKNESS)

            # Cleanup
            inactive = {k for k in active_alerts if int(k.split('-')[1]) not in current_ids}
            active_alerts -= inactive
            for tid in list(track_history.keys()):
                if tid not in current_ids: del track_history[tid]
                
            if len(live_alerts) > 50: live_alerts[:] = live_alerts[:50]

        ok, buf = cv2.imencode(".jpg", frame)
        if ok: yield(b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n")
    
    cap.release()
    
@app.route('/')
def index(): return render_template('index.html', cameras=CAMERAS)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def alerts(): return jsonify(live_alerts)

@app.route('/history')
def history():
    page = int(request.args.get('page', 1))
    alerts_paginated = AlertLog.query.order_by(AlertLog.timestamp.desc()).paginate(page=page, per_page=50)
    return render_template('history.html', alerts_pagination=alerts_paginated)

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    print("[INFO] Server starting...")
    app.run(debug=True, port=5000, host='127.0.0.1', threaded=True, use_reloader=False)