import cv2
import numpy as np
import cv2.aruco as aruco
from path_utils import load_path_points, find_grid_path, draw_grid_path, draw_grid


def main():
    # Initialize the webcam (or use a video file path)
    cap = cv2.VideoCapture(1)

    # 1. Setup the ArUco Dictionary and Parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    LOOKAHEAD_DISTANCE = 36
    GRID_SIZE = 48
    REACH_THRESHOLD = GRID_SIZE

    # Load path and track which waypoint the robot is heading toward
    waypoints = load_path_points()
    current_wp_index = 0

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

                # Calculate Front Midpoint (top edge is "forward")
                front_x = int((c[0][0] + c[1][0]) / 2)
                front_y = int((c[0][1] + c[1][1]) / 2)
                top_mid_x = int((c[0][0] + c[1][0]) / 2)
                top_mid_y = int((c[0][1] + c[1][1]) / 2)
                top_mid_pt = (top_mid_x, top_mid_y)

                # Heading Vector
                dx = front_x - center_x
                dy = front_y - center_y
                magnitude = np.hypot(dx, dy)

                if magnitude > 0:
                    norm_dx = dx / magnitude
                    norm_dy = dy / magnitude

                    lookahead_x = int(top_mid_x + (norm_dx * LOOKAHEAD_DISTANCE))
                    lookahead_y = int(top_mid_y + (norm_dy * LOOKAHEAD_DISTANCE))
                    lookahead_pt = (lookahead_x, lookahead_y)

                    cv2.line(frame, top_mid_pt, lookahead_pt, (0, 255, 0), 4)
                    cv2.circle(frame, lookahead_pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"({lookahead_x}, {lookahead_y})", (lookahead_x + 10, lookahead_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw grid guide toward current waypoint
                if current_wp_index < len(waypoints):
                    target_pt = waypoints[current_wp_index]
                    dist_to_target = np.hypot(center_x - target_pt[0], center_y - target_pt[1])

                    if dist_to_target < REACH_THRESHOLD:
                        current_wp_index += 1

                    if current_wp_index < len(waypoints):
                        target_pt = waypoints[current_wp_index]
                        grid_path = find_grid_path(center_pt, target_pt, grid_size=GRID_SIZE)
                        if grid_path:
                            draw_grid_path(frame, grid_path, num_blocks=2)
                        cv2.circle(frame, target_pt, 10, (0, 165, 255), -1)
                        cv2.putText(frame, f"WP{current_wp_index}", (target_pt[0] + 12, target_pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "PATH COMPLETE", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        draw_grid(frame)
        cv2.imshow("Path Follower", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
