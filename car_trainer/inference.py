"""
inference.py  –  DAVE2 inference utilities for the end-to-end robot.

Provides:
  • InferenceEngine   – loads the model once; predict from raw OpenCV frames.
  • run_inference_loop – generator that pulls frames from a queue and yields
                         (steering, throttle) pairs; drop-in for the robot
                         controller's main loop.

Standalone usage:
  python inference.py <image_path> [weights_path]
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path fix so this module can be imported from robot_controller/ too
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model import DAVE2  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = os.path.join(_THIS_DIR, "dave2_robot_model.pth")
IMG_H, IMG_W = 66, 200          # DAVE-2 input size (height, width)
STEERING_SCALE = 50.0           # inverse of dataset normalisation (angle / 50)
DEFAULT_THROTTLE = 28           # constant throttle used in AI mode (0-100)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    Loads a DAVE2 model once and exposes fast per-frame prediction.

    Parameters
    ----------
    weights_path : str
        Path to the .pth state-dict file.
    throttle : int
        Fixed throttle value (0-100) sent alongside the predicted steering.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).

    Usage
    -----
    engine = InferenceEngine()
    steering, throttle = engine.predict_frame(bgr_frame)
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS,
        throttle: int = DEFAULT_THROTTLE,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.throttle = throttle

        self.model = DAVE2().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[InferenceEngine] Loaded weights from '{weights_path}' on {self.device}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _preprocess(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """
        Mirror dataset.py preprocessing:
          BGR → RGB → resize (200, 66) → float32 → CHW tensor → batch dim.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_W, IMG_H))           # (width, height)
        arr = resized.astype(np.float32)                    # 0-255, matches training
        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)                               # HWC → CHW
            .unsqueeze(0)                                   # → (1, 3, 66, 200)
            .to(self.device)
        )
        return tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_frame(self, bgr_frame: np.ndarray) -> tuple[int, int]:
        """
        Run a single forward pass on a raw OpenCV BGR frame.

        Returns
        -------
        (steering, throttle) : tuple[int, int]
            steering  – integer in roughly [-50, 50] (raw units used by robot).
            throttle  – fixed throttle set at construction time.
        """
        tensor = self._preprocess(bgr_frame)
        with torch.no_grad():
            raw = self.model(tensor).item()         # tanh output in [-1, 1]
        steering = int(round(raw * STEERING_SCALE))
        return steering, self.throttle

    def predict_frame_raw(self, bgr_frame: np.ndarray) -> float:
        """Return the raw tanh output in [-1, 1] without scaling."""
        tensor = self._preprocess(bgr_frame)
        with torch.no_grad():
            return self.model(tensor).item()


# ---------------------------------------------------------------------------
# run_inference_loop  –  generator for robot_controller integration
# ---------------------------------------------------------------------------
def run_inference_loop(
    frame_queue,
    engine: InferenceEngine,
    stop_event,
):
    """
    Generator that consumes frames from *frame_queue* and yields predictions.

    Design mirrors the camera_thread / main loop pattern in the robot
    controller so it can be embedded directly in the main loop:

        for steering, throttle in run_inference_loop(frame_queue, engine, stop_event):
            # send to robot ...
            if stop_event.is_set():
                break

    Parameters
    ----------
    frame_queue : queue.Queue
        The same queue populated by camera_thread.  Frames are consumed
        non-destructively (the generator calls get_nowait and puts the frame
        back if the caller might still need it – see *consume* param).
    engine : InferenceEngine
        A pre-loaded InferenceEngine instance.
    stop_event : threading.Event
        Shared stop signal; the generator exits when it is set.

    Yields
    ------
    (steering, throttle, bgr_frame) : tuple[int, int, np.ndarray]
        Includes the frame so the main loop can still display / record it.
    """
    from queue import Empty

    while not stop_event.is_set():
        try:
            frame = frame_queue.get_nowait()
        except Empty:
            # No frame yet – yield None so the caller can still tick
            yield None, None, None
            continue

        steering, throttle = engine.predict_frame(frame)
        yield steering, throttle, frame


# ---------------------------------------------------------------------------
# Legacy single-image helpers (kept for backwards compatibility / testing)
# ---------------------------------------------------------------------------
def load_model(weights_path: str = DEFAULT_WEIGHTS, device: torch.device | None = None) -> DAVE2:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAVE2().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict_steering_from_path(image_path: str, weights_path: str = DEFAULT_WEIGHTS) -> float:
    """Load model & predict from a file path. Matches dataset.py preprocessing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.device = device
    engine.model = model
    engine.throttle = DEFAULT_THROTTLE
    raw = engine.predict_frame_raw(bgr)
    steering = int(round(raw * STEERING_SCALE))
    print(f"Predicted steering (raw={raw:.4f})  →  {steering}  [range ≈ -50 … +50]")
    return raw


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [weights_path]")
        print("Example: python inference.py ../data/frame_001.jpg dave2_robot_model.pth")
        sys.exit(1)

    _image_path = sys.argv[1]
    _weights = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WEIGHTS
    predict_steering_from_path(_image_path, _weights)