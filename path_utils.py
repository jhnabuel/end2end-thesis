import cv2
import numpy as np
from collections import deque


def get_first_path_point():
    with open('path.txt', 'r') as f:
        line = f.readline().strip()
        x, y = map(int, line.split(','))
        return (x, y)


def load_path_points():
    """Load all waypoints from path.txt."""
    points = []
    with open('path.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(int, line.split(','))
                points.append((x, y))
    return points


def find_grid_path(start_px, target_px, grid_size=48):
    """BFS from start pixel to the grid cell containing target pixel, moving through adjacent grids."""
    start_cell = (start_px[0] // grid_size, start_px[1] // grid_size)
    target_cell = (target_px[0] // grid_size, target_px[1] // grid_size)

    if start_cell == target_cell:
        return [start_px, target_px]

    queue = deque()
    queue.append(start_cell)
    visited = {start_cell: None}

    # 4-connected neighbors (right, left, down, up)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        current = queue.popleft()
        if current == target_cell:
            break
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if neighbor not in visited and neighbor[0] >= 0 and neighbor[1] >= 0:
                visited[neighbor] = current
                queue.append(neighbor)

    # Reconstruct path
    if target_cell not in visited:
        return []

    path_cells = []
    cell = target_cell
    while cell is not None:
        path_cells.append(cell)
        cell = visited[cell]
    path_cells.reverse()

    # Convert grid cells to center pixel coordinates
    path_pts = []
    for c in path_cells:
        cx = c[0] * grid_size + grid_size // 2
        cy = c[1] * grid_size + grid_size // 2
        path_pts.append((cx, cy))
    return path_pts


def draw_grid_path(frame, grid_path, num_blocks=3):
    """Draw only the first num_blocks points of the grid path as a guide."""
    visible = grid_path[:num_blocks + 1]  # show up to num_blocks segments
    for i in range(len(visible) - 1):
        cv2.line(frame, visible[i], visible[i + 1], (255, 255, 0), 3)
        cv2.circle(frame, visible[i], 5, (255, 0, 255), -1)
    if visible:
        cv2.circle(frame, visible[-1], 5, (255, 0, 255), -1)


def draw_grid(frame, grid_size=48):
    height, width, _ = frame.shape
    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)


def draw_path(frame, path_points):
    with open('path.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = map(int, line.strip().split(','))
            path_points.append((x, y))
        for i in range(len(path_points) - 1):
            cv2.line(frame, path_points[i], path_points[i + 1], (0, 0, 255), 2)
