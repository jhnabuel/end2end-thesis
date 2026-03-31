import cv2
import numpy as np

DEST_PTS = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
ARENA_SIZE = (800, 800)


class ArenaWarper:
    def __init__(self):
        self.matrix = None

    def generate_arena(self, frame, corners, ids):
        if ids is None:
            if self.matrix is None:
                return frame
            return cv2.warpPerspective(frame, self.matrix, ARENA_SIZE)

        topLeftBig = topRightBig = bottomLeftBig = bottomRightBig = None
        flat_ids = ids.flatten()

        for markerCorner, markerID in zip(corners, flat_ids):
            (topLeft, topRight, bottomRight,
             bottomLeft) = markerCorner.reshape((4, 2))
            mcX = (topLeft[0] + bottomRight[0]) / 2.0
            mcY = (topLeft[1] + bottomRight[1]) / 2.0
            if markerID == 24:
                topLeftBig = [mcX, mcY]
            elif markerID == 42:
                topRightBig = [mcX, mcY]
            elif markerID == 66:
                bottomLeftBig = [mcX, mcY]
            elif markerID == 70:
                bottomRightBig = [mcX, mcY]

        if None in (topLeftBig, topRightBig, bottomLeftBig, bottomRightBig):
            if self.matrix is None:
                return frame
        else:
            src = np.float32(
                [topLeftBig, topRightBig, bottomLeftBig, bottomRightBig])
            self.matrix = cv2.getPerspectiveTransform(src, DEST_PTS)

        return cv2.warpPerspective(frame, self.matrix, ARENA_SIZE)
