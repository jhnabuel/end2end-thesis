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


def project_onto_polyline(px, py, polyline):
    """Project point (px,py) onto a polyline. Returns (segment_index, t, proj_x, proj_y, cumulative_dist)."""
    best_dist = float('inf')
    best_seg = 0
    best_t = 0.0
    best_proj = polyline[0]

    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        abx, aby = bx - ax, by - ay
        ab_len2 = abx * abx + aby * aby
        if ab_len2 == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len2))
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        d = np.hypot(px - proj_x, py - proj_y)
        if d < best_dist:
            best_dist = d
            best_seg = i
            best_t = t
            best_proj = (proj_x, proj_y)

    # Cumulative distance up to projection point
    cum = 0.0
    for i in range(best_seg):
        cum += np.hypot(polyline[i + 1][0] - polyline[i][0],
                        polyline[i + 1][1] - polyline[i][1])
    cum += np.hypot(best_proj[0] - polyline[best_seg][0],
                    best_proj[1] - polyline[best_seg][1])

    return best_seg, best_t, best_proj[0], best_proj[1], cum


def sample_polyline_ahead(polyline, start_dist, length, num_points=20):
    """Sample num_points evenly spaced along polyline from start_dist to start_dist+length."""
    # Build cumulative distances
    cum_dists = [0.0]
    for i in range(len(polyline) - 1):
        cum_dists.append(cum_dists[-1] + np.hypot(
            polyline[i + 1][0] - polyline[i][0],
            polyline[i + 1][1] - polyline[i][1]))
    total_len = cum_dists[-1]

    end_dist = min(start_dist + length, total_len)
    if end_dist <= start_dist:
        return []

    points = []
    for k in range(num_points + 1):
        d = start_dist + (end_dist - start_dist) * k / num_points
        # Find which segment this distance falls on
        for i in range(len(cum_dists) - 1):
            if cum_dists[i + 1] >= d:
                seg_len = cum_dists[i + 1] - cum_dists[i]
                if seg_len > 0:
                    t = (d - cum_dists[i]) / seg_len
                else:
                    t = 0.0
                x = polyline[i][0] + t * (polyline[i + 1][0] - polyline[i][0])
                y = polyline[i][1] + t * (polyline[i + 1][1] - polyline[i][1])
                points.append((int(x), int(y)))
                break
    return points


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
    
    
    
    # Build pixel polyline from cell centers
    path_polyline = [cell_center(*c, GRID_SIZE) for c in path_cells]
    PATH_VIEW_LENGTH = GRID_SIZE * 2  # how far ahead (pixels) the "driver" can see
    
    cv2.namedWindow("Path Follower", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path Follower", 1400, 1400) # Set a specific width and height
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution set to: {width}x{height}")
    print(f"Path loaded : {len(path_cells)} grid cells")
    print(path_polyline)
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
    
                    
                # --- Project car onto path polyline for smooth tracking ---
                seg, t, proj_x, proj_y, car_dist = project_onto_polyline(
                    center_x, center_y, path_polyline)
                current_wp_index = seg

                # Sample the visible portion of the road (from start to car + view length)
                total_visible_length = car_dist + PATH_VIEW_LENGTH
                num_samples = max(2, int(total_visible_length / 4)) # Sample every 4 pixels for smooth track
                visible_pts = sample_polyline_ahead(
                    path_polyline, 0, total_visible_length, num_points=num_samples)
                
                car_to_path_dist = np.hypot(center_x - int(proj_x), center_y - int(proj_y))
                # print(f"car distance to path: {car_to_path_dist}")
                if car_to_path_dist < GRID_SIZE / 2:
                    
                    if len(visible_pts) >= 2:
                        ROAD_WIDTH = int(GRID_SIZE * 0.50)
                        
                        # Create a black overlay for the track
                        overlay = np.zeros_like(frame)
                        pts_arr = np.array(visible_pts, np.int32)
                        
                        # Draw a thick polyline on the overlay to naturally handle all inner/outer corners
                        cv2.polylines(overlay, [pts_arr], isClosed=False, color=(0, 165, 255), thickness=ROAD_WIDTH)
                        
                        # Find the outline of the track
                        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Draw the clean continuous boundary outline onto the frame
                        cv2.drawContours(frame, contours, -1, (0, 165, 255), 1)
                        
                        # Blended fill for the track directly from the overlay
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


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


