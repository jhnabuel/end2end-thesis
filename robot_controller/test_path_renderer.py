import cv2.aruco as aruco
from path_renderer import PathRenderer
from path_utils import load_path_points
from robot_aoi import ArenaWarper
import cv2
import numpy as np
import pickle

ARENA_IDS = {24, 42, 66, 70}
CAR_ID = 0

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

def main_path_renderer():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    GRID_SIZE = 64
    with open("calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    camera_matrix = calib["mtx"]
    dist_coeffs = calib["dist"]

    shared_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    shared_detector = aruco.ArucoDetector(shared_dict, aruco.DetectorParameters())
    path_polyline = load_path_polyline(GRID_SIZE)
    if len(path_polyline) < 2:
        print("ERROR: Need at least 2 path points. Draw a path first.")
        cap.release()
        return

    renderer = PathRenderer(path_polyline=path_polyline, detector=shared_detector, grid_size=GRID_SIZE)
    warper = ArenaWarper()

    cv2.namedWindow("Path View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path View", 1000, 1000)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # --- Arena detection (on raw frame) ---
        arena_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_corners, detected_ids, _ = shared_detector.detectMarkers(arena_gray)

        arena_corners_filtered, arena_ids_list = [], []
        if detected_ids is not None:
            for corner, mid in zip(detected_corners, detected_ids.flatten()):
                if mid in ARENA_IDS:
                    arena_corners_filtered.append(corner)
                    arena_ids_list.append([mid])

        arena_ids_arr = np.array(arena_ids_list) if arena_ids_list else None
        warped = warper.generate_arena(frame, arena_corners_filtered, arena_ids_arr)  

        # --- Car detection (on warped frame) ---
        car_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        detected_car_corners, detected_car_ids, _ = shared_detector.detectMarkers(car_gray) 

        car_corners_filtered, car_ids_list = [], []
        if detected_car_ids is not None:
            for corner, mid in zip(detected_car_corners, detected_car_ids.flatten()):
                if mid == CAR_ID:
                    car_corners_filtered.append(corner)
                    car_ids_list.append([mid])

        car_ids_arr = np.array(car_ids_list) if car_ids_list else None

        out, _ = renderer.generate_cnn_frame(warped, predetected=(car_corners_filtered, car_ids_arr))
        renderer.draw_debug(out)
        cv2.imshow("Path View", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        yield out

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for _ in main_path_renderer():
        pass