import cv2
import numpy as np
import os
import pickle
from robot_aoi import ArenaWarper
import cv2.aruco as aruco

ARENA_IDS = {24, 42, 66, 70}
GRID_SIZE = 64
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "Path Editor"

highlighted_cells = []
cell_set = set()
warper = ArenaWarper()

shared_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
shared_detector = aruco.ArucoDetector(shared_dict, aruco.DetectorParameters())

def pixel_to_cell(x, y):
    return (x // GRID_SIZE, y // GRID_SIZE)

def cell_center(col, row):
    return (col * GRID_SIZE + GRID_SIZE // 2, row * grid_size + GRID_SIZE // 2)

def cell_center(col, row):
    return (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cell = pixel_to_cell(x, y)
        if cell not in cell_set:
            highlighted_cells.append(cell)
            cell_set.add(cell)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if highlighted_cells:
            last = highlighted_cells.pop()
            cell_set.discard(last)

def detect_and_warp(frame):
    arena_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_corners, detected_ids, _ = shared_detector.detectMarkers(arena_gray)

    arena_corners_filtered, arena_ids_list = [], []
    if detected_ids is not None:
        for corner, mid in zip(detected_corners, detected_ids.flatten()):
            if mid in ARENA_IDS:
                arena_corners_filtered.append(corner)
                arena_ids_list.append([mid])

    arena_ids_arr = np.array(arena_ids_list) if arena_ids_list else None
    return warper.generate_arena(frame, arena_corners_filtered, arena_ids_arr)

def draw_overlays(warped):
    frame = warped.copy()
    height, width = frame.shape[:2]

    for col, row in highlighted_cells:
        top_left = (col * GRID_SIZE, row * GRID_SIZE)
        bottom_right = ((col + 1) * GRID_SIZE, (row + 1) * GRID_SIZE)
        cv2.rectangle(frame, top_left, bottom_right, (0, 200, 0), -1)

    for i in range(len(highlighted_cells) - 1):
        pt1 = cell_center(*highlighted_cells[i])
        pt2 = cell_center(*highlighted_cells[i + 1])
        cv2.line(frame, pt1, pt2, (0, 255, 100), 2)

    for idx, cell in enumerate(highlighted_cells):
        cx, cy = cell_center(*cell)
        cv2.putText(frame, str(idx + 1), (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for x in range(0, width, GRID_SIZE):
        cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(0, height, GRID_SIZE):
        cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)

    cv2.putText(frame, f"Waypoints: {len(highlighted_cells)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "L-Click: add  |  R-Click: undo  |  Q: save & quit",
                (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame

def save_path():
    with open("path.txt", "w") as f:
        for col, row in highlighted_cells:
            cx, cy = cell_center(col, row)
            f.write(f"{cx},{cy}\n")
    print(f"[path_editor] Saved {len(highlighted_cells)} waypoints to path.txt")

def load_existing_path():
    if not os.path.exists("path.txt"):
        return
    try:
        with open("path.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    x, y = map(int, line.split(","))
                    cell = pixel_to_cell(x, y)
                    if cell not in cell_set:
                        highlighted_cells.append(cell)
                        cell_set.add(cell)
        print(f"[path_editor] Loaded {len(highlighted_cells)} existing waypoints.")
    except Exception as e:
        print(f"[path_editor] Could not load path.txt: {e}")

def main():
    load_existing_path()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 900)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("[path_editor] Window open. Left-click to add waypoints, right-click to undo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        warped = detect_and_warp(frame)
        display = draw_overlays(warped)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    save_path()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()