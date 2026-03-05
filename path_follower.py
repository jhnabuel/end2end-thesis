import cv2
import numpy as np
import cv2.aruco as aruco
from path_utils import load_path_points, find_grid_path, draw_grid_path, draw_grid, load_path_line
<<<<<<< HEAD
=======

def pixel_to_cell(x, y, grid_size):
    """Convert pixel coords to grid cell (col, row)."""
    return (x // grid_size, y // grid_size)


def cell_center(col, row, grid_size):
    """Return pixel center of a grid cell."""
    return (col * grid_size + grid_size // 2,
            row * grid_size + grid_size // 2)
>>>>>>> d53861ab51f65b0d973535c5f7ce44241e95fea7


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
<<<<<<< HEAD
    REACH_THRESHOLD = GRID_SIZE
=======
>>>>>>> d53861ab51f65b0d973535c5f7ce44241e95fea7

    # Load path and track which waypoint the robot is heading toward
    raw_waypoints = load_path_points()
    path_cells = [pixel_to_cell(x,y, GRID_SIZE) for x, y, in raw_waypoints]
    path_cells = [c for i, c in enumerate(path_cells) if i == 0 or c != path_cells[i-1]]

    current_wp_index = 0
<<<<<<< HEAD
=======

>>>>>>> d53861ab51f65b0d973535c5f7ce44241e95fea7
    cv2.namedWindow("Path Follower", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path Follower", 1400, 1400) # Set a specific width and height
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution set to: {width}x{height}")
<<<<<<< HEAD
=======
    print(f"Path loaded : {len(path_cells)} grid cells")
>>>>>>> d53861ab51f65b0d973535c5f7ce44241e95fea7
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
                
                if path_cells:
                    while current_wp_index < len(path_cells) and car_cell == path_cells[current_wp_index]:
                        current_wp_index += 1
                    
                    if current_wp_index < len(path_cells):
                        target_cell = path_cells[current_wp_index]
                        target_pt = cell_center(*target_cell, GRID_SIZE)
        
                        current_cell_pt = cell_center(*car_cell, GRID_SIZE)
                        cv2.line(frame, current_cell_pt, target_pt, (0, 165, 255), 2)
                        cv2.circle(frame, current_cell_pt, 5, (0, 165, 255), -1)
                        cv2.circle(frame, target_pt, 5, (0, 165, 255), -1)
                else:
                    cv2.putText(frame, "PATH COMPLETE", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
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


if __name__ == "__main__":
    main()
