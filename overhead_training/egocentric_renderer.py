import cv2
import numpy as np
import math
from path_renderer import PathRenderer

class EgocentricRenderer:
    def __init__(self, path_renderer: PathRenderer, crop_size=200):
        self.path_renderer = path_renderer
        self.crop_size = crop_size
        self.half_crop = crop_size // 2

    def process_egocentric_frame(self, raw_frame):
        """Rotate and crop the augmented frame around the car so it faces up."""
        global_frame, c = self.path_renderer.generate_cnn_frame(raw_frame)

        if c is None:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        center_x = int(np.mean(c[:, 0]))
        center_y = int(np.mean(c[:, 1]))

        # Heading from ArUco top edge midpoint
        front_x = (c[0][0] + c[1][0]) / 2.0
        front_y = (c[0][1] + c[1][1]) / 2.0
        angle_deg = math.degrees(math.atan2(front_y - center_y, front_x - center_x))

        # Rotate so the car faces up, output directly as a crop_size x crop_size image
        M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg + 90, 1.0)
        M[0, 2] += self.half_crop - center_x
        M[1, 2] += self.half_crop - center_y
        return cv2.warpAffine(global_frame, M, (self.crop_size, self.crop_size))
