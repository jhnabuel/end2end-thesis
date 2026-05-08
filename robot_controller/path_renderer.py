#path_renderer.py
import cv2
import numpy as np
import cv2.aruco as aruco
import math


def project_onto_path(px, py, path):
    """Project point (px,py) onto the polyline. Returns (arc_length, distance_to_path, projected_point)."""
    best_dist = float('inf')
    best_arc = 0.0
    best_pt = path[0].astype(np.float64)
    best_tangent = (path[1] - path[0]).astype(np.float64)   
    cumulative = 0.0

    car = np.array([px, py], dtype=np.float64)

    for i in range(len(path) - 1):
        a = path[i].astype(np.float64)
        b = path[i + 1].astype(np.float64)
        ab = b - a
        seg_len = np.linalg.norm(ab)
        if seg_len < 1e-6:
            cumulative += seg_len
            continue
        
        
        t = np.clip(np.dot(car - a, ab) / (seg_len * seg_len), 0.0, 1.0)
        proj = a + t * ab
        d_sq = float(np.dot(car - proj, car - proj))
        
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_arc     = cumulative + t * seg_len
            best_pt      = proj
            best_tangent = ab

        # Signed CTE via cross product: tangent × (car − nearest_point)
        # positive → car is to the RIGHT of the path direction
        # negative → car is to the LEFT
        tx, ty = float(best_tangent[0]), float(best_tangent[1])
        ex, ey = px - best_pt[0], py - best_pt[1]
        seg_len = math.sqrt(tx * tx + ty * ty + 1e-9)
        signed_cte = (tx * ey - ty * ex) / seg_len

        cumulative += seg_len

    return best_arc, signed_cte, best_pt, best_tangent


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

        t0 = max(0.0, (arc_start - seg_start) /
                 seg_len) if seg_len > 1e-6 else 0.0
        t1 = min(1.0, (arc_end - seg_start) /
                 seg_len) if seg_len > 1e-6 else 1.0

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


def _build_arc_index(smooth_pts):
    """Build a cumulative arc-length array for the precomputed smooth path."""
    diffs = np.diff(smooth_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc_index = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return arc_index


class PathRenderer:
    def __init__(self, path_polyline, detector, grid_size=44, road_color=(255, 0, 0),
                 forward_px=200, backward_px=50):
        self.path_polyline = np.array(path_polyline, dtype=np.float64)
        self.grid_size = grid_size
        self.track_thickness = int(grid_size * 1)
        self.road_color = road_color
        self.forward_px = forward_px
        self.backward_px = backward_px
        self.target_id = 0

        # Precompute total arc length (on the original non-wrapped path)
        diffs = np.diff(self.path_polyline, axis=0)
        self.total_arc = np.sum(np.linalg.norm(diffs, axis=1))

        # --- Precompute smoothed path ---
        dense_pts = sample_path_by_arc(
            self.path_polyline, 0.0, self.total_arc, spacing=2.0)
        self.smooth_pts = chaikin_smooth(dense_pts, iterations=3)
        self.smooth_arc_index = _build_arc_index(self.smooth_pts)
        self.smooth_pts_int = np.int32(np.round(self.smooth_pts))

        # --- Build a WRAPPED version of the smooth path for lookahead ---
        # Append one extra copy so arc math can continue past the seam.
        # We only need forward_px worth of wrap, but one full copy is safe.
        self.smooth_pts_wrapped = np.concatenate(
            [self.smooth_pts, self.smooth_pts], axis=0)
        self.smooth_arc_wrapped = _build_arc_index(self.smooth_pts_wrapped)
        self.smooth_pts_wrapped_int = np.int32(np.round(self.smooth_pts_wrapped))
        # --------------------------------

        # ArUco detection
        self.detector = detector
        self.last_corners = None
        self.frames_lost = 0

    def _slice_smooth_path(self, arc_start, arc_end):
        """
        Return smooth points between arc_start and arc_end, wrapping around
        the loop seam if arc_end > total_arc.
        """
        if arc_end <= self.total_arc:
            # Normal case: no wrap needed, use the plain index
            idx_start = np.searchsorted(
                self.smooth_arc_index, arc_start, side='left')
            idx_end = np.searchsorted(
                self.smooth_arc_index, arc_end, side='right')
            idx_start = max(0, idx_start - 1)
            idx_end = min(len(self.smooth_pts_int), idx_end + 1)
            return self.smooth_pts_int[idx_start:idx_end]
        else:
            # Wrap case: slice from arc_start to end-of-loop, then 0 to remainder
            remainder = arc_end - self.total_arc

            # Part 1: arc_start → total_arc
            idx_start = np.searchsorted(
                self.smooth_arc_index, arc_start, side='left')
            idx_start = max(0, idx_start - 1)
            part1 = self.smooth_pts_int[idx_start:]

            # Part 2: 0 → remainder (beginning of the loop)
            idx_end = np.searchsorted(
                self.smooth_arc_index, remainder, side='right')
            idx_end = min(len(self.smooth_pts_int), idx_end + 1)
            part2 = self.smooth_pts_int[:idx_end]

            if len(part1) == 0:
                return part2
            if len(part2) == 0:
                return part1
            return np.concatenate([part1, part2], axis=0)

    def generate_cnn_frame(self, frame, predetected=None, draw_lookahead=True, black_bg = False, ):

        _empty_metrics = {
            'cte': 0.0, 'heading_error': 0.0,
            'car_arc': 0.0, 'on_path': False,
        }

        output = np.zeros_like(frame) if black_bg else frame.copy()

        #  --- ArUco detection ---
        if predetected is not None:
            corners, ids = predetected
            ids = ids if (ids is not None and len(ids) > 0) else None
        else:
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detector.detectMarkers(gray)[:2]

        c = None
        if ids is None:
            self.frames_lost += 1
            if self.last_corners is None or self.frames_lost > 5:
                self.last_corners = None
                return output, None, _empty_metrics
            c = self.last_corners
        else:
            matched = False
            for i in range(len(ids)):
                if ids[i][0] == self.target_id:
                    c = corners[i][0]
                    self.last_corners = c
                    self.frames_lost = 0
                    matched = True
                    break
            if not matched:
                self.frames_lost += 1
                if self.last_corners is None or self.frames_lost > 5:
                    return output, None, _empty_metrics
                c = self.last_corners

         # --- Car centre and heading (from ArUco corners) ---
        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))

        if not black_bg:    
            cv2.polylines(frame, [np.int32(self.path_polyline)], False, (128, 128, 128),
                        self.track_thickness, cv2.LINE_8)
            cv2.addWeighted(frame, 0.9, output, 0.6, 0, output)
        else:
            cv2.polylines(output, [np.int32(self.path_polyline)], False, (80, 80, 80),
                  self.track_thickness, cv2.LINE_8)

        # Heading from ArUco top edge
        top_mid_x    = (c[2][0] + c[3][0]) / 2.0
        top_mid_y    = (c[2][1] + c[3][1]) / 2.0
        bottom_mid_x = (c[0][0] + c[1][0]) / 2.0
        bottom_mid_y = (c[0][1] + c[1][1]) / 2.0

        # Car heading angle in warped-frame space
        angle = math.atan2(top_mid_y - bottom_mid_y, top_mid_x - bottom_mid_x)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # --- Project car onto path → CTE + heading error ---
        car_arc, signed_cte, _, tangent = project_onto_path(cx, cy, self.path_polyline)
        raw_dist = abs(signed_cte)
        on_path = raw_dist < self.grid_size

        # Heading Error: difference between car angle and path tangle angle
        path_angle    = math.atan2(float(tangent[1]), float(tangent[0]))
        heading_error = angle - path_angle
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        metrics = {
            'cte':           signed_cte if on_path else 0.0,
            'heading_error': heading_error if on_path else 0.0,
            'car_arc':       car_arc,
            'on_path':       on_path,
        }

        if on_path:
            # path tangent 
            idx = np.searchsorted(self.smooth_arc_index, car_arc)
            idx = np.clip(idx, 0, len(self.smooth_pts_int) - 6)
            p1 = self.smooth_pts_int[idx]
            p2 = self.smooth_pts_int[idx + 5]
            path_dx = float(p2[0] - p1[0])
            path_dy = float(p2[1] - p1[1])
            dot   = cos_a * path_dx + sin_a * path_dy
        
            if dot >= 0:
                # Clockwise — normal case
                arc_start = max(0.0, car_arc - self.backward_px)
                arc_end = car_arc + self.forward_px
                slice_pts = self._slice_smooth_path(arc_start, arc_end)
            else:
                # Counter-clockwise — forward/backward are flipped
                raw_start = car_arc - self.forward_px
                raw_end = min(self.total_arc, car_arc + self.backward_px)

                if raw_start >= 0:
                    slice_pts = self._slice_smooth_path(raw_start, raw_end)
                else:
                    # raw_start is negative, wrap it to end of loop
                    wrapped_start = self.total_arc + raw_start
                    part1 = self._slice_smooth_path(wrapped_start, self.total_arc)
                    part2 = self._slice_smooth_path(0.0, raw_end)
                    if len(part1) > 0 and len(part2) > 0:
                        slice_pts = np.concatenate([part1, part2], axis=0)
                    elif len(part1) > 0:
                        slice_pts = part1
                    else:
                        slice_pts = part2

            road_color = (255, 255, 255) if black_bg else self.road_color
            if len(slice_pts) >= 2:
                cv2.polylines(output, [slice_pts], False, road_color,
                              self.track_thickness // 2, cv2.LINE_AA)


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
            cv2.line(output, (cx, cy), (lx, ly), (0, 255, 0), 4, cv2.LINE_8)

        cv2.polylines(output, [np.int32(self.path_polyline)], False, (0, 255, 255),
              max(2, self.track_thickness // 30
                  ), cv2.LINE_8)
        
        return output, c, metrics

    def draw_debug(self, frame):
        if self.last_corners is not None:
            c = self.last_corners
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            car_arc, signed_cte, _, _ = project_onto_path(cx, cy, self.path_polyline)
            cv2.putText(
                frame,
                f"car:({cx},{cy}) arc={car_arc:.0f}/{self.total_arc:.0f} "
                f"cte={signed_cte:.1f}px",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
        return frame