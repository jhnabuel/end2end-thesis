import cv2
import sys
import os
import pickle

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
os.chdir(parent_dir)

from overhead_training.robot_aoi import generate_arena
from path_utils import load_path_points
from path_renderer import PathRenderer
import cv2.aruco as aruco

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    GRID_SIZE = 64
    with open("calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    camera_matrix = calib["mtx"]
    dist_coeffs = calib["dist"]

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

    cv2.namedWindow("Path View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path View", 1000, 1000)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = arena_detector.detectMarkers(gray)
        warped = generate_arena(frame, corners, ids)
        out, _ = renderer.generate_cnn_frame(warped)
        renderer.draw_debug(out)
        cv2.imshow("Path View", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
