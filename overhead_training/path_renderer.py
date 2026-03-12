import cv2
import numpy as np
import cv2.aruco as aruco
import math


def project_onto_path(px, py, path):
    """Project point (px,py) onto the polyline. Returns (arc_length, distance_to_path, projected_point)."""
    best_dist = float('inf')
    best_arc = 0.0
    best_pt = path[0].astype(np.float64)
    cumulative = 0.0

    for i in range(len(path) - 1):
        a = path[i].astype(np.float64)
        b = path[i + 1].astype(np.float64)
        ab = b - a
        seg_len = np.linalg.norm(ab)
        if seg_len < 1e-6:
            cumulative += seg_len
            continue
        t = np.clip(np.dot(np.array([px, py], dtype=np.float64) - a, ab) / (seg_len * seg_len), 0.0, 1.0)
        proj = a + t * ab
        d = np.linalg.norm(np.array([px, py]) - proj)
        if d < best_dist:
            best_dist = d
            best_arc = cumulative + t * seg_len
            best_pt = proj
        cumulative += seg_len

    return best_arc, best_dist, best_pt


def sample_path_by_arc(path, arc_start, arc_end, spacing=2.0):
    """Sample evenly-spaced points along the polyline between two arc-length values."""
    points = []
    cumulative = 0.0

    for i in range(len(path) - 1):
        a = path[i].astype(np.float64)
        b = path[i + 1].astype(np.float64)
        seg_len = np.linalg.norm(b - a)
        seg_start = cumulative
        seg_end = cumulative + seg_len

        if seg_end < arc_start or seg_start > arc_end:
            cumulative += seg_len
            continue

        # Clip to the requested range
        t0 = max(0.0, (arc_start - seg_start) / seg_len) if seg_len > 1e-6 else 0.0
        t1 = min(1.0, (arc_end - seg_start) / seg_len) if seg_len > 1e-6 else 1.0

        local_start = t0 * seg_len
        local_end = t1 * seg_len
        t = local_start
        while t <= local_end:
            pt = a + (t / seg_len) * (b - a) if seg_len > 1e-6 else a
            points.append(pt)
            t += spacing

        cumulative += seg_len

    return np.array(points) if points else np.empty((0, 2))


def chaikin_smooth(pts, iterations=3):
    """Chaikin's corner-cutting: iteratively replace corners with smooth curves."""
    for _ in range(iterations):
        if len(pts) < 2:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        smoothed = np.empty((2 * len(q), 2), dtype=pts.dtype)
        smoothed[0::2] = q
        smoothed[1::2] = r
        pts = smoothed
    return pts


class PathRenderer:
    def __init__(self, path_polyline, grid_size=44, road_color=(255, 255, 255),
                 forward_px=200, backward_px=80):
        self.path_polyline = np.array(path_polyline, dtype=np.float64)
        self.grid_size = grid_size
        self.track_thickness = int(grid_size * 0.70)
        self.road_color = road_color
        self.forward_px = forward_px
        self.backward_px = backward_px

        # Precompute total arc length
        diffs = np.diff(self.path_polyline, axis=0)
        self.total_arc = np.sum(np.linalg.norm(diffs, axis=1))

        # ArUco detection
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

        # State for surviving brief detection drops
        self.last_corners = None
        self.frames_lost = 0

    def generate_cnn_frame(self, frame, draw_lookahead=True):
        """Track the car, render nearby path segments and a fixed-size car polygon.
        Returns (augmented_frame, corners) where corners is None if the car is lost."""
        output = frame.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            self.frames_lost += 1
            if self.last_corners is None or self.frames_lost > 5:
                self.last_corners = None
                return output, None
            c = self.last_corners
        else:
            c = corners[0][0]
            self.last_corners = c
            self.frames_lost = 0

        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))

        # Draw path segment sliding smoothly with the car
        car_arc, dist, _ = project_onto_path(cx, cy, self.path_polyline)
        if dist < self.grid_size:
            arc_start = max(0.0, car_arc - self.backward_px)
            arc_end = min(self.total_arc, car_arc + self.forward_px)
            pts = sample_path_by_arc(self.path_polyline, arc_start, arc_end)
            if len(pts) >= 2:
                smooth_pts = np.int32(np.round(chaikin_smooth(pts)))
                cv2.polylines(output, [smooth_pts], False, self.road_color,
                              self.track_thickness, cv2.LINE_AA)

        # Heading from ArUco top edge
        front_x = (c[0][0] + c[1][0]) / 2.0
        front_y = (c[0][1] + c[1][1]) / 2.0
        angle = math.atan2(front_y - cy, front_x - cx)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Draw fixed-size car polygon
        hw = 15
        car_pts = np.int32([[
            (cx + hw*cos_a - hw*sin_a, cy + hw*sin_a + hw*cos_a),
            (cx + hw*cos_a + hw*sin_a, cy + hw*sin_a - hw*cos_a),
            (cx - hw*cos_a + hw*sin_a, cy - hw*sin_a - hw*cos_a),
            (cx - hw*cos_a - hw*sin_a, cy - hw*sin_a + hw*cos_a),
        ]])
        cv2.fillPoly(output, car_pts, (0, 0, 255), cv2.LINE_AA)

        if draw_lookahead:
            lx = int(cx + cos_a * 35)
            ly = int(cy + sin_a * 35)
            cv2.line(output, (cx, cy), (lx, ly), (0, 255, 0), 4, cv2.LINE_AA)

        return output, c

    def draw_debug(self, frame):
        """Draw waypoint markers with indices and car coordinates."""
        for i, (wx, wy) in enumerate(self.path_polyline):
            cv2.circle(frame, (int(wx), int(wy)), 4, (255, 0, 255), -1)
            cv2.putText(frame, f"{i}:({int(wx)},{int(wy)})", (int(wx)+5, int(wy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

        if self.last_corners is not None:
            c = self.last_corners
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            car_arc, dist, _ = project_onto_path(cx, cy, self.path_polyline)
            cv2.putText(frame, f"car:({cx},{cy}) arc={car_arc:.0f}/{self.total_arc:.0f} d={dist:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return frame
