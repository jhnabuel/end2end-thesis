import cv2
import math
import numpy as np

from path_renderer import PathRenderer


class EgocentricRenderer:
    def __init__(self, path_renderer: PathRenderer, crop_size: int = 200):
        self.path_renderer = path_renderer
        self.crop_size = int(crop_size)
        self.half_crop = self.crop_size // 2

    def process_egocentric_frame(self, raw_frame, predetected=None, black_bg: bool = True):
        """Render BEV overlays, then rotate/crop so the car faces up."""

        _empty_metrics = {
            'cte': 0.0, 'heading_error': 0.0,
            'car_arc': 0.0, 'on_path': False,
        }

        frame, corners, metrics = self.path_renderer.generate_cnn_frame(
            raw_frame, predetected=predetected, black_bg=black_bg
        )
        if corners is None:
            return None, None, _empty_metrics

        center_x = float(np.mean(corners[:, 0]))
        center_y = float(np.mean(corners[:, 1]))

        top_mid_x = (corners[2][0] + corners[3][0]) / 2.0
        top_mid_y = (corners[2][1] + corners[3][1]) / 2.0
        bottom_mid_x = (corners[0][0] + corners[1][0]) / 2.0
        bottom_mid_y = (corners[0][1] + corners[1][1]) / 2.0

        angle_deg = math.degrees(
            math.atan2(top_mid_y - bottom_mid_y, top_mid_x - bottom_mid_x)
        )

        M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg + 90.0, 1.0)
        M[0, 2] += self.half_crop - center_x
        M[1, 2] += self.half_crop - center_y
        ego = cv2.warpAffine(frame, M, (self.crop_size, self.crop_size))

        return ego, corners,metrics
