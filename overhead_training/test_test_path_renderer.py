import cv2
import sys
import os
import pickle
import time
import threading

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
os.chdir(parent_dir)

from overhead_training.robot_aoi import generate_arena
from path_utils import load_path_points
from path_renderer import PathRenderer
import cv2.aruco as aruco
from flask import Flask, Response, render_template_string

app = Flask(__name__)

#streaming globals
output_frame = None
lock = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Path View</title></head>
<body style="background:#111;display:flex;justify-content:center;align-items:center;height:100vh;margin:0">
    <img src="/video_feed" style="max-width:100%;max-height:100vh"/>
</body>
</html>
"""


def pixel_to_cell(x, y, grid_size):
    return (x // grid_size, y // grid_size)

def cell_center(col, row, grid_size):
    return (col * grid_size + grid_size // 2, row * grid_size + grid_size // 2)

def load_path_polyline(grid_size):
    raw = load_path_points()
    if not raw:
        return []
    cells = [pixel_to_cell(x, y, grid_size) for x, y in raw]
    cells = [c for i, c in enumerate(cells) if i == 0 or c != cells[i-1]]
    return [cell_center(*c, grid_size) for c in cells]

def main():
    global output_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    with open("calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    camera_matrix = calib["mtx"]
    dist_coeffs = calib["dist"]

    GRID_SIZE = 64

    # Arena corners detector - 5x5 markers
    arena_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    arena_detector = aruco.ArucoDetector(arena_dict, aruco.DetectorParameters())

    # Car marker detector - 4x4 markers (separate!)
    car_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    car_detector = aruco.ArucoDetector(car_dict, aruco.DetectorParameters())

    path_polyline = load_path_polyline(GRID_SIZE)
    if len(path_polyline) < 2:
        print("ERROR: Need at least 2 path points. Draw a path first.")
        cap.release()
        return

    # Pass car_detector, NOT arena_detector
    renderer = PathRenderer(path_polyline=path_polyline, detector=car_detector, grid_size=GRID_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, retrying...")
            time.sleep(0.2)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = arena_detector.detectMarkers(gray)
        warped = generate_arena(frame, corners, ids)

        if warped is None:
            display = frame
        else:
            out, _ = renderer.generate_cnn_frame(warped)
            renderer.draw_debug(out)
            display = out
        with lock:
            output_frame = display.copy()

        #cap.release()

def generateMjpeg():
    global output_frame
    while True:
        with lock:
            if output_frame is not None:
                local_frame = output_frame.copy()
            else:
                local_frame = None
            
        if local_frame is not None:
            ret, buffer = cv2.imencode(".jpg", local_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if ret:
                yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )   
                
        else:
            time.sleep(0.2)
            continue
    
        time.sleep(0.3)
        

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    return Response(
        generateMjpeg(),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    t = threading.Thread(target=main, daemon=True)
    t.start()
    app.run(host = "0.0.0.0", port = 5000, debug = False)
