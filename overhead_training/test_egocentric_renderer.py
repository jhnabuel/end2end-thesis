import cv2
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
os.chdir(parent_dir)

from path_utils import load_path_points
from path_renderer import PathRenderer
from egocentric_renderer import EgocentricRenderer


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
    cap = cv2.VideoCapture(1)
    GRID_SIZE = 64

    path_polyline = load_path_polyline(GRID_SIZE)
    if len(path_polyline) < 2:
        print("ERROR: Need at least 2 path points. Draw a path first.")
        cap.release()
        return

    renderer = PathRenderer(path_polyline=path_polyline, grid_size=GRID_SIZE)
    ego_renderer = EgocentricRenderer(path_renderer=renderer, crop_size=200)

    cv2.namedWindow("Egocentric View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Egocentric View", 600, 600)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ego_frame = ego_renderer.process_egocentric_frame(frame)
        cv2.imshow("Egocentric View", ego_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
