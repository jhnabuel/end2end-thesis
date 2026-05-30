"""
run_logger.py — structured evaluation-run logging for the egocentric CNN driving thesis.

Encapsulates everything the data-collection redo needs (see
data-collection-redo/04_logging_audit_and_plan.md):

  * Run identity / metadata block (track, scenario, model, trial, start pose, ...)
  * One telemetry row per NEW camera frame (caller decides when a frame is new)
  * Outcome + success capture (marker-lost streak, boundary breach, completion)
  * Lap detection (start/finish region re-entry)
  * Structured output path: metrics/runs/<track_id>/<model_id>/trial_<NN>.json

Per-frame and summary statistics (RMSE, P95, per-lap, etc.) are intentionally NOT
computed here — they are derived offline by the batch metrics script from the rows
this logger writes, keeping the real-time loop lightweight.

This module has no hardware or OpenCV dependencies, so it can be unit-tested off-robot.
"""

import os
import json
import math
import time


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
RUNS_DIR = os.path.join(_REPO_ROOT, "metrics", "runs")
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "run_config.json")


def load_run_config(path=DEFAULT_CONFIG):
    """Load the per-run metadata template. Returns a dict with safe defaults if missing."""
    defaults = {
        "track_id": "unknown_track",
        "scenario": 0,
        "is_held_out": False,
        "model_id": "unknown_model",
        "operator": "",
        "battery_v": None,
        "loop_hz_target": 20,
        "px_per_cm": None,
        "lane_half_width_px": 40.0,   # used for boundary-breach detection
        "marker_lost_fail_frames": 30,  # consecutive lost frames => failure
        "lap_start_radius_px": 30.0,    # re-entry radius around path start for lap counting
        "notes": "",
    }
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user = json.load(fh)
        defaults.update({k: user[k] for k in user})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def _next_trial_index(track_id, model_id):
    """Scan the output dir and return the next free trial number (1-based)."""
    out_dir = os.path.join(RUNS_DIR, track_id, model_id)
    if not os.path.isdir(out_dir):
        return 1
    max_idx = 0
    for name in os.listdir(out_dir):
        if name.startswith("trial_") and name.endswith(".json"):
            try:
                max_idx = max(max_idx, int(name[len("trial_"):-len(".json")]))
            except ValueError:
                pass
    return max_idx + 1


class RunLogger:
    """Accumulates one evaluation run and writes a structured JSON on finalize()."""

    def __init__(self, config=None, start_pose=None, path_start_xy=None):
        self.cfg = config if config is not None else load_run_config()
        self.trial_idx = _next_trial_index(self.cfg["track_id"], self.cfg["model_id"])
        self.path_start_xy = path_start_xy  # (x, y) of path[0] in warped px, for lap detection

        self.frames = []
        self.start_wall_ms = int(time.time() * 1000)

        # Outcome trackers
        self._marker_lost_streak = 0
        self._max_marker_lost_streak = 0
        self._boundary_breached = False
        self._max_abs_cte = 0.0
        self._max_seg_idx = 0
        self._n_path_points = None

        # Lap detection state
        self._laps = 0
        self._left_start_zone = False

        self.meta = {
            "track_id": self.cfg["track_id"],
            "scenario": self.cfg["scenario"],
            "is_held_out": self.cfg["is_held_out"],
            "model_id": self.cfg["model_id"],
            "trial_idx": self.trial_idx,
            "start_pose": start_pose,
            "battery_v": self.cfg["battery_v"],
            "operator": self.cfg["operator"],
            "loop_hz_target": self.cfg["loop_hz_target"],
            "px_per_cm": self.cfg["px_per_cm"],
            "notes": self.cfg["notes"],
        }

    # ----------------------------------------------------------------------
    def _update_laps(self, cx, cy):
        if self.path_start_xy is None or cx is None or cy is None:
            return
        sx, sy = self.path_start_xy
        d = math.hypot(cx - sx, cy - sy)
        r = self.cfg["lap_start_radius_px"]
        if d > r * 2.0:
            self._left_start_zone = True
        elif d <= r and self._left_start_zone:
            self._laps += 1
            self._left_start_zone = False

    # ----------------------------------------------------------------------
    def log_frame(self, *, frame_seq, timestamp_ms, cte_px, heading_error,
                  steering, speed, marker_detected,
                  ai_steering_raw=None, cx=None, cy=None,
                  nearest_x=None, nearest_y=None, seg_idx=0, n_path_points=None,
                  path_start_xy=None):
        """Append one telemetry row. Call ONLY when a new camera frame arrived."""
        # Capture the path start once (enables lap detection without constructor wiring)
        if self.path_start_xy is None and path_start_xy is not None:
            self.path_start_xy = path_start_xy

        # dt from previous row
        dt_ms = None
        if self.frames:
            dt_ms = timestamp_ms - self.frames[-1]["timestamp_ms"]

        # Outcome trackers
        if marker_detected:
            self._marker_lost_streak = 0
        else:
            self._marker_lost_streak += 1
            self._max_marker_lost_streak = max(
                self._max_marker_lost_streak, self._marker_lost_streak)

        abs_cte = abs(cte_px)
        self._max_abs_cte = max(self._max_abs_cte, abs_cte)
        if abs_cte > self.cfg["lane_half_width_px"]:
            self._boundary_breached = True
        self._max_seg_idx = max(self._max_seg_idx, seg_idx)
        if n_path_points:
            self._n_path_points = n_path_points

        self._update_laps(cx, cy)

        self.frames.append({
            "frame_seq": frame_seq,
            "lap_index": self._laps,
            "timestamp_ms": timestamp_ms,
            "dt_ms": dt_ms,
            "cx": cx, "cy": cy,
            "nearest_x": nearest_x, "nearest_y": nearest_y,
            "seg_idx": seg_idx,
            "cte_px": round(cte_px, 4),
            "heading_error": round(heading_error, 4),
            "heading_deg": round(math.degrees(heading_error), 2),
            "ai_steering_raw": ai_steering_raw,
            "steering": steering,
            "speed": speed,
            "marker_detected": marker_detected,
        })

    # ----------------------------------------------------------------------
    @property
    def auto_failed(self):
        """True if a failure condition has already been met (caller may early-stop)."""
        return (self._boundary_breached
                or self._marker_lost_streak >= self.cfg["marker_lost_fail_frames"])

    def _completion_fraction(self):
        if self._n_path_points and self._n_path_points > 1:
            return round(min(1.0, self._max_seg_idx / (self._n_path_points - 1)), 4)
        return None

    def _measured_hz(self):
        if len(self.frames) < 2:
            return None
        span_s = (self.frames[-1]["timestamp_ms"] - self.frames[0]["timestamp_ms"]) / 1000.0
        return round((len(self.frames) - 1) / span_s, 2) if span_s > 0 else None

    # ----------------------------------------------------------------------
    def finalize(self, result=None, failure_reason=None):
        """
        Write the run JSON. `result` may be 'success'/'fail'/None.
        If None, it is inferred: auto-fail conditions => 'fail', else 'success'.
        Returns the output path (or None if nothing was logged).
        """
        if not self.frames:
            return None

        if result is None:
            if self._boundary_breached:
                result, failure_reason = "fail", failure_reason or "boundary_breach"
            elif self._marker_lost_streak >= self.cfg["marker_lost_fail_frames"]:
                result, failure_reason = "fail", failure_reason or "marker_lost"
            else:
                result = "success"

        duration_s = round(
            (self.frames[-1]["timestamp_ms"] - self.frames[0]["timestamp_ms"]) / 1000.0, 3)

        payload = {
            "meta": self.meta,
            "outcome": {
                "result": result,
                "failure_reason": failure_reason,
                "frames_survived": len(self.frames),
                "completion_fraction": self._completion_fraction(),
                "laps_completed": self._laps,
                "duration_s": duration_s,
                "measured_hz": self._measured_hz(),
                "max_abs_cte_px": round(self._max_abs_cte, 4),
                "max_marker_lost_streak": self._max_marker_lost_streak,
            },
            "frames": self.frames,
        }

        out_dir = os.path.join(RUNS_DIR, self.meta["track_id"], self.meta["model_id"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"trial_{self.trial_idx:02d}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return out_path
