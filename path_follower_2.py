import cv2
import numpy as np
import cv2.aruco as aruco
from path_utils import load_path_points, find_grid_path, draw_grid_path, draw_grid, load_path_line

def pixel_to_cell(x, y, grid_size):
    """Convert pixel coords to grid cell (col, row)."""
    return (x // grid_size, y // grid_size)
    
def cell_to_pixel(x, y, grid_size):
    """Convert grid cell to top-left pixel position."""
    return (x * grid_size, y * grid_size)

def cell_center(col, row, grid_size):
    """Return pixel center of a grid cell."""
    return (col * grid_size + grid_size // 2,
            row * grid_size + grid_size // 2)


def main():
    # Initialize the webcam (or use a video file path)
    cap = cv2.VideoCapture(0)

    # 1. Setup the ArUco Dictionary and Parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    LOOKAHEAD_DISTANCE = 36
    GRID_SIZE = 44
    # Load path and track which waypoint the robot is heading toward
    raw_waypoints = load_path_points()
    path_cells = [pixel_to_cell(x,y, GRID_SIZE) for x, y, in raw_waypoints]
    path_cells = [c for i, c in enumerate(path_cells) if i == 0 or c != path_cells[i-1]]
    
    current_wp_index = 0
    
    cv2.namedWindow("Path Follower", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path Follower", 1400, 1400) # Set a specific width and height
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution set to: {width}x{height}")
    print(f"Path loaded : {len(path_cells)} grid cells")
    print(path_cells)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                c = corners[i][0]

                # Calculate Center Point
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                center_pt = (center_x, center_y)
                car_cell = pixel_to_cell(center_x, center_y, GRID_SIZE)

                # Calculate Front Midpoint (top edge is "forward")
                front_x = int((c[0][0] + c[1][0]) / 2)
                front_y = int((c[0][1] + c[1][1]) / 2)
                top_mid_pt = (front_x, front_y)

                # Heading Vector
                dx = front_x - center_x
                dy = front_y - center_y
                magnitude = np.hypot(dx, dy)

                if magnitude > 0:
                    norm_dx = dx / magnitude
                    norm_dy = dy / magnitude

                    lookahead_x = int(front_x + (norm_dx * LOOKAHEAD_DISTANCE))
                    lookahead_y = int(front_y + (norm_dy * LOOKAHEAD_DISTANCE))
                    lookahead_pt = (lookahead_x, lookahead_y)

                    cv2.line(frame, top_mid_pt, lookahead_pt, (0, 255, 0), 4)
                    cv2.circle(frame, lookahead_pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"({lookahead_x}, {lookahead_y})", (lookahead_x + 10, lookahead_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
                    
                target_offset = 2  # change this dynamically

                if car_cell in path_cells:
                    i = path_cells.index(car_cell)

                    if i + target_offset < len(path_cells):

                        current_cell_pt = center_pt

                        # draw path progressively
                        prev_pt = current_cell_pt

                        for step in range(1, target_offset + 1):
                            next_cell = path_cells[i + step]
                            next_pt = cell_center(*next_cell, GRID_SIZE)

                            cv2.line(frame, prev_pt, next_pt, (0, 165, 255), 2)
                            cv2.circle(frame, next_pt, 5, (0, 165, 255), -1)

                            prev_pt = next_pt
                            
                            
                        cv2.circle(frame, current_cell_pt, 5, (0, 165, 255), -1)
                    else:
                        next_value = None

 


                cv2.putText(frame, f"Car cell: {car_cell}",
                            (8, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
        draw_grid(frame,GRID_SIZE)
        load_path_line(frame)
        
        
        cv2.imshow("Path Follower", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#def frame2

if __name__ == "__main__":
    main()


