# path_renderer.py  — optimised version
#
# Key changes vs original:
#   1. sample_path_by_arc spacing: 2.0 → 6.0  (3x fewer points everywhere)
#   2. Static road pre-rendered to self._road_cache at init; no polylines/
#      addWeighted every frame.
#   3. project_onto_path accepts an arc_index + hint_arc so it only searches
#      a local window (~300 px) instead of the full smooth path.
#   4. generate_cnn_frame passes the previous arc result as the hint so the
#      search window is always tight.

import cv2
import numpy as np
import cv2.aruco as aruco
import math


# ---------------------------------------------------------------------------
# Path geometry helpers
# ---------------------------------------------------------------------------

def project_onto_path(px, py, path, arc_index=None, hint_arc=None,
                      search_radius=200.0):
    """
    Project point (px, py) onto the polyline.

    Returns (arc_length, signed_cte, projected_point, tangent_vector).

    If arc_index and hint_arc are supplied the search is restricted to a
    ±search_radius window around hint_arc, making this O(window/spacing)
    instead of O(total_path_length).
    """
    car = np.array([px, py], dtype=np.float64)

    # --- determine which segment slice to search ---
    if arc_index is not None and hint_arc is not None:
        lo = int(np.searchsorted(arc_index,
                                 max(0.0, hint_arc - search_radius),
                                 side='left'))
        hi = int(np.searchsorted(arc_index,
                                 hint_arc + search_radius,
                                 side='right'))
        lo = max(0, lo - 1)
        hi = min(len(path) - 1, hi + 1)
        search_path   = path[lo:hi + 1]
        offset_arc    = arc_index[lo]
    else:
        search_path = path
        lo           = 0
        offset_arc   = 0.0

    best_dist_sq  = float('inf')
    best_arc      = offset_arc
    best_pt       = search_path[0].astype(np.float64)
    best_tangent  = (search_path[1] - search_path[0]).astype(np.float64) \
                    if len(search_path) > 1 else np.array([1.0, 0.0])
    cumulative    = offset_arc

    for i in range(len(search_path) - 1):
        a = search_path[i].astype(np.float64)
        b = search_path[i + 1].astype(np.float64)
        ab = b - a
        seg_len = np.linalg.norm(ab)
        if seg_len < 1e-6:
            cumulative += seg_len
            continue

        t    = np.clip(np.dot(car - a, ab) / (seg_len * seg_len), 0.0, 1.0)
        proj = a + t * ab
        d_sq = float(np.dot(car - proj, car - proj))

        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_arc     = cumulative + t * seg_len
            best_pt      = proj
            best_tangent = ab

        cumulative += seg_len

    # Signed CTE via cross product
    tx, ty = float(best_tangent[0]), float(best_tangent[1])
    ex, ey = px - best_pt[0], py - best_pt[1]
    seg_len_safe = math.sqrt(tx * tx + ty * ty + 1e-9)
    signed_cte   = (tx * ey - ty * ex) / seg_len_safe

    return best_arc, signed_cte, best_pt, best_tangent


def sample_path_by_arc(path, arc_start, arc_end, spacing=6.0):
    """Sample evenly-spaced points along the polyline between two arc-length values."""
    points     = []
    cumulative = 0.0

    for i in range(len(path) - 1):
        a       = path[i].astype(np.float64)
        b       = path[i + 1].astype(np.float64)
        seg_len = np.linalg.norm(b - a)
        seg_start = cumulative
        seg_end   = cumulative + seg_len

        if seg_end < arc_start or seg_start > arc_end:
            cumulative += seg_len
            continue

        t0 = max(0.0, (arc_start - seg_start) / seg_len) if seg_len > 1e-6 else 0.0
        t1 = min(1.0, (arc_end   - seg_start) / seg_len) if seg_len > 1e-6 else 1.0

        local_start = t0 * seg_len
        local_end   = t1 * seg_len
        t = local_start
        while t <= local_end:
            pt = a + (t / seg_len) * (b - a) if seg_len > 1e-6 else a
            points.append(pt)
            t += spacing

        cumulative += seg_len

    return np.array(points) if points else np.empty((0, 2))


def bezier_corners(path, num_pts=20):
    if len(path) < 3:
        return path

    is_closed = np.linalg.norm(path[0] - path[-1]) < 1e-6
    p = path[:-1] if is_closed else path
    n = len(p)

    corners = set()
    for i in range(n):
        if not is_closed and (i == 0 or i == n - 1):
            continue
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        A, B, C  = p[prev_idx], p[i], p[next_idx]
        v1, v2   = B - A, C - B
        l1, l2   = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 > 1e-6 and l2 > 1e-6 and np.dot(v1, v2) / (l1 * l2) < 0.99:
            corners.add(i)

    new_path = []
    consumed = set()

    for i in range(n):
        if i in consumed:
            continue

        if i in corners:
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            A = p[prev_idx].astype(np.float64)
            B = p[i].astype(np.float64)
            C = p[next_idx].astype(np.float64)

            t      = np.linspace(0, 1, num_pts)[:, np.newaxis]
            bezier = (1 - t) ** 2 * A + 2 * (1 - t) * t * B + t ** 2 * C

            if len(new_path) > 0 and np.linalg.norm(new_path[-1] - A) < 1e-6:
                new_path.pop()

            for pt in bezier:
                if len(new_path) == 0 or np.linalg.norm(new_path[-1] - pt) > 1e-6:
                    new_path.append(pt)

            consumed.add(next_idx)
        else:
            pt = p[i]
            if len(new_path) == 0 or np.linalg.norm(new_path[-1] - pt) > 1e-6:
                new_path.append(pt)

    if is_closed and len(new_path) > 0:
        if np.linalg.norm(new_path[0] - new_path[-1]) > 1e-6:
            new_path.append(new_path[0])

    return np.array(new_path)


def _build_arc_index(smooth_pts):
    """Build a cumulative arc-length array for the precomputed smooth path."""
    diffs    = np.diff(smooth_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


# ---------------------------------------------------------------------------
# PathRenderer
# ---------------------------------------------------------------------------

class PathRenderer:
    # Spacing between densified smooth-path samples (pixels).
    # 6 px is visually identical to 2 px but produces 3× fewer points,
    # cutting polyline draw time and projection search time proportionally.
    SMOOTH_SPACING = 6.0

    def __init__(self, path_polyline, detector, grid_size=44,
                 road_color=(255, 0, 0), forward_px=200, backward_px=50,
                 arena_size=(800, 800)):
        self.path_polyline = np.array(path_polyline, dtype=np.float64)
        self.grid_size     = grid_size
        self.track_thickness = int(grid_size * 1)
        self.road_color    = road_color
        self.forward_px    = forward_px
        self.backward_px   = backward_px
        self.target_id     = 0
        self.arena_size    = arena_size   # (width, height) of warped frame

        # Precompute total arc length (original waypoints)
        diffs           = np.diff(self.path_polyline, axis=0)
        self.total_arc  = np.sum(np.linalg.norm(diffs, axis=1))

        # --- Smooth path (Bezier corners + dense resampling) ---
        bezier_path = bezier_corners(self.path_polyline, num_pts=20)
        diffs_bez   = np.diff(bezier_path, axis=0)
        total_bez   = np.sum(np.linalg.norm(diffs_bez, axis=1))

        dense_pts = sample_path_by_arc(
            bezier_path, 0.0, total_bez, spacing=self.SMOOTH_SPACING)

        self.smooth_pts       = dense_pts
        self.smooth_arc_index = _build_arc_index(self.smooth_pts)
        self.smooth_pts_int   = np.int32(np.round(self.smooth_pts))

        # Smooth path becomes the official polyline for projection
        self.path_polyline = self.smooth_pts
        self.total_arc     = self.smooth_arc_index[-1]

        # Wrapped version for forward-lookahead past the loop seam
        self.smooth_pts_wrapped     = np.concatenate(
            [self.smooth_pts, self.smooth_pts], axis=0)
        self.smooth_arc_wrapped     = _build_arc_index(self.smooth_pts_wrapped)
        self.smooth_pts_wrapped_int = np.int32(np.round(self.smooth_pts_wrapped))

        # --- Pre-render static road to a cache image ---
        # This replaces the two cv2.polylines calls that previously ran every
        # frame.  generate_cnn_frame just copies / blends this cache instead.
        w, h = self.arena_size
        self._road_cache_black = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.polylines(self._road_cache_black, [self.smooth_pts_int], False,
                      (80, 80, 80), self.track_thickness, cv2.LINE_8)
        cv2.polylines(self._road_cache_black, [self.smooth_pts_int], False,
                      (0, 255, 255), max(2, self.track_thickness // 30),
                      cv2.LINE_8)

        # For the non-black-bg path we need just the gray road strip so we
        # can addWeighted it onto the real frame once.
        self._road_strip_gray = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.polylines(self._road_strip_gray, [self.smooth_pts_int], False,
                      (128, 128, 128), self.track_thickness, cv2.LINE_8)

        # ArUco
        self.detector     = detector
        self.last_corners = None
        self.frames_lost  = 0

        # Cache the last successful arc projection so the next frame can use
        # a localised search window.
        self._last_car_arc = None

        # Driving direction: True = forward (increasing arc), False = reverse.
        # Determined from arc displacement between frames — reliable at corners
        # unlike the heading/tangent dot product.
        self._driving_forward = True

        # Cache the last rendered output frame so callers (e.g. ego_renderer)
        # can retrieve it without triggering a second full render pass.
        self.last_out = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slice_smooth_path(self, arc_start, arc_end):
        """
        Return smooth-path int points between arc_start and arc_end,
        wrapping around the loop seam when arc_end > total_arc.
        """
        smooth_total = self.smooth_arc_index[-1]

        if arc_end <= smooth_total:
            idx_start = np.searchsorted(
                self.smooth_arc_index, arc_start, side='left')
            idx_end   = min(len(self.smooth_pts_int),
                            np.searchsorted(self.smooth_arc_index,
                                            arc_end, side='right') + 1)
            return self.smooth_pts_int[idx_start:idx_end]
        else:
            remainder = arc_end - smooth_total
            idx_start = max(0, np.searchsorted(
                self.smooth_arc_index, arc_start, side='left') - 1)
            part1 = self.smooth_pts_int[idx_start:]
            idx_end = min(len(self.smooth_pts_int),
                          np.searchsorted(self.smooth_arc_index,
                                          remainder, side='right') + 1)
            part2 = self.smooth_pts_int[:idx_end]
            if len(part1) == 0:
                return part2
            if len(part2) == 0:
                return part1
            return np.concatenate([part1, part2], axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cnn_frame(self, frame, predetected=None,
                           draw_lookahead=True, black_bg=False):

        _empty_metrics = {
            'cte': 0.0, 'heading_error': 0.0,
            'car_arc': 0.0, 'on_path': False,
            'cx': None, 'cy': None,
            'nearest_x': None, 'nearest_y': None,
            'seg_idx': 0, 'n_path_points': None,
            'path_start': None,
        }

        # --- Guard: frame must match arena cache dimensions ---
        # ArenaWarper returns the raw camera frame (640×480) when it hasn't
        # found all 4 corner markers yet (matrix is None).  The road cache is
        # built for arena_size (800×800), so addWeighted would crash.
        # Return the frame as-is and wait for the homography to stabilise.
        cache_h, cache_w = self._road_cache_black.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if frame_h != cache_h or frame_w != cache_w:
            self.last_out = frame.copy()
            return self.last_out, None, _empty_metrics

        # --- Build output from pre-rendered road cache ---
        if black_bg:
            # Fast path: just copy the cached black canvas
            output = self._road_cache_black.copy()
        else:
            # Blend the static gray road strip onto the real frame once
            output = cv2.addWeighted(frame, 0.9,
                                     self._road_strip_gray, 0.6, 0)

        # --- ArUco detection ---
        if predetected is not None:
            corners, ids = predetected
            ids = ids if (ids is not None and len(ids) > 0) else None
        else:
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detector.detectMarkers(gray)[:2]

        c = None
        if ids is None:
            self.frames_lost += 1
            if self.last_corners is None or self.frames_lost > 5:
                self.last_corners     = None
                self._last_car_arc    = None   # force full-path search on re-acquisition
                self._driving_forward = True   # reset direction until re-established
                self.last_out = output
                return output, None, _empty_metrics
            c = self.last_corners
        else:
            matched = False
            for i in range(len(ids)):
                if ids[i][0] == self.target_id:
                    c               = corners[i][0]
                    self.last_corners = c
                    self.frames_lost  = 0
                    matched           = True
                    break
            if not matched:
                self.frames_lost += 1
                if self.last_corners is None or self.frames_lost > 5:
                    self.last_corners     = None
                    self._last_car_arc    = None   # force full-path search on re-acquisition
                    self._driving_forward = True   # reset direction until re-established
                    self.last_out = output
                    return output, None, _empty_metrics
                c = self.last_corners

        # --- Car centre and heading ---
        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))

        top_mid_x    = (c[0][0] + c[1][0]) / 2.0
        top_mid_y    = (c[0][1] + c[1][1]) / 2.0
        bottom_mid_x = (c[2][0] + c[3][0]) / 2.0
        bottom_mid_y = (c[2][1] + c[3][1]) / 2.0

        angle        = math.atan2(top_mid_y - bottom_mid_y,
                                  top_mid_x - bottom_mid_x)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # --- Project car onto path (localised search) ---
        # Use a wider search window when we have no prior arc hint, or when
        # the car has been lost for several frames (hint may have drifted).
        _search_r = 400.0 if (self._last_car_arc is None or self.frames_lost > 0) \
                    else 250.0

        # Seam-safe projection: when the hint is within search_radius of the
        # loop seam (arc=0 / arc=total_arc), a windowed search on the linear
        # smooth_pts array misses the wrap-around portion.  In that case we
        # search on smooth_pts_wrapped (two copies concatenated) with the hint
        # offset into the centre of the first copy, then map the result back.
        hint = self._last_car_arc
        near_seam = (hint is not None and
                     (hint < _search_r or hint > self.total_arc - _search_r))

        if near_seam:
            # Shift hint into the middle of the wrapped array so the window
            # straddles the seam naturally.
            wrapped_hint = hint + self.total_arc
            car_arc_wrapped, signed_cte, proj_pt, tangent = project_onto_path(
                cx, cy,
                self.smooth_pts_wrapped,
                arc_index     = self.smooth_arc_wrapped,
                hint_arc      = wrapped_hint,
                search_radius = _search_r,
            )
            # Map result back to [0, total_arc)
            car_arc_raw = car_arc_wrapped % self.total_arc
        else:
            car_arc_raw, signed_cte, proj_pt, tangent = project_onto_path(
                cx, cy,
                self.path_polyline,
                arc_index     = self.smooth_arc_index,
                hint_arc      = hint,
                search_radius = _search_r,
            )
        # --- Driving direction from arc displacement (stable across corners) ---
        if self._last_car_arc is not None:
            arc_delta = car_arc_raw - self._last_car_arc
            # Wrap-around correction: if the car crossed the seam the raw delta
            # jumps by ~total_arc in the wrong direction.
            half_arc = self.total_arc / 2.0
            if arc_delta > half_arc:
                arc_delta -= self.total_arc
            elif arc_delta < -half_arc:
                arc_delta += self.total_arc
            # Only update direction when the car has actually moved enough to
            # be meaningful (filter out sub-pixel jitter from ArUco noise).
            if abs(arc_delta) > 2.0:
                self._driving_forward = arc_delta >= 0.0

        self._last_car_arc = car_arc_raw   # cache for next frame

        on_path = abs(signed_cte) < self.grid_size

        path_angle    = math.atan2(float(tangent[1]), float(tangent[0]))
        heading_error = angle - path_angle
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # Some ArUco orientations can be flipped by 180°; choose the heading
        # direction that produces the smaller signed error.
        heading_error_alt = heading_error + math.pi
        heading_error_alt = (heading_error_alt + math.pi) % (2 * math.pi) - math.pi
        if abs(heading_error_alt) < abs(heading_error):
            heading_error = heading_error_alt

        metrics = {
            'cte':           signed_cte,
            'heading_error': heading_error,
            'car_arc':       car_arc_raw,
            'on_path':       on_path,
            # --- Position + progress surfaced for run logging ---
            # (lap detection, completion %, trajectory plots).  Progress is
            # expressed as arc-length so completion = car_arc / total_arc; the
            # run logger reads it as seg_idx / (n_path_points - 1).
            'cx':            cx,
            'cy':            cy,
            'nearest_x':     float(proj_pt[0]),
            'nearest_y':     float(proj_pt[1]),
            'seg_idx':       int(round(car_arc_raw)),
            'n_path_points': int(self.total_arc) + 1,
            'path_start':    (float(self.path_polyline[0][0]),
                              float(self.path_polyline[0][1])),
        }

        if on_path:
            # Lookahead direction follows actual driving direction (determined
            # from arc displacement, not heading — stable through corners).
            if self._driving_forward:
                arc_start = max(0.0, car_arc_raw - self.backward_px)
                arc_end   = car_arc_raw + self.forward_px
            else:
                arc_start = max(0.0, car_arc_raw - self.forward_px)
                arc_end   = car_arc_raw + self.backward_px
            slice_pts = self._slice_smooth_path(arc_start, arc_end)

            road_color = (255, 255, 255) if black_bg else self.road_color
            if len(slice_pts) >= 2:
                cv2.polylines(output, [slice_pts], False, road_color,
                              self.track_thickness // 2, cv2.LINE_AA)

        # --- Car polygon ---
        hw     = 15
        car_pts = np.int32([[
            (cx + hw * cos_a - hw * sin_a, cy + hw * sin_a + hw * cos_a),
            (cx + hw * cos_a + hw * sin_a, cy + hw * sin_a - hw * cos_a),
            (cx - hw * cos_a + hw * sin_a, cy - hw * sin_a - hw * cos_a),
            (cx - hw * cos_a - hw * sin_a, cy - hw * sin_a + hw * cos_a),
        ]])
        cv2.fillPoly(output, car_pts, (0, 0, 255), cv2.LINE_AA)

        if draw_lookahead:
            lx = int(cx + cos_a * 35)
            ly = int(cy + sin_a * 35)
            cv2.line(output, (cx, cy), (lx, ly), (0, 255, 0), 4, cv2.LINE_8)

        # The cyan centreline is already in the road cache; skip re-drawing it.
        # (For non-black-bg it was also drawn in the original, so we add it back
        # here only for that case to keep visual parity.)
        if not black_bg:
            cv2.polylines(output, [self.smooth_pts_int], False, (0, 255, 255),
                          max(2, self.track_thickness // 30), cv2.LINE_8)

        self.last_out = output
        return output, c, metrics

    def draw_debug(self, frame):
        if self.last_corners is not None:
            c   = self.last_corners
            cx  = int(np.mean(c[:, 0]))
            cy  = int(np.mean(c[:, 1]))
            car_arc_raw, signed_cte, _, _ = project_onto_path(
                cx, cy,
                self.path_polyline,
                arc_index     = self.smooth_arc_index,
                hint_arc      = self._last_car_arc,
                search_radius = 250.0,
            )
            cv2.putText(
                frame,
                f"car:({cx},{cy}) arc={car_arc_raw:.0f}/{self.total_arc:.0f} "
                f"cte={signed_cte:.1f}px",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA,
            )
        return frame