import cv2
import numpy as np

def generate_arena(frame, corners, ids):
    topLeftBig = None
    topRightBig = None
    bottomLeftBig = None
    bottomRightBig = None

    if ids is None:
        return frame

    flat_ids = ids.flatten()
    for (markerCorner, markerID) in zip(corners, flat_ids):
        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
        mcX = int((topLeft[0] + bottomRight[0]) / 2.0)
        mcY = int((topLeft[1] + bottomRight[1]) / 2.0)
        match markerID:
            case 24:
                topLeftBig = [mcX, mcY]
            case 42:
                topRightBig = [mcX, mcY]
            case 66:
                bottomLeftBig = [mcX, mcY]
            case 70:
                bottomRightBig = [mcX, mcY]

    if None in (topLeftBig, topRightBig, bottomLeftBig, bottomRightBig):
        return frame
    
    sourcePts = np.float32([topLeftBig, topRightBig, bottomLeftBig, bottomRightBig])
    destPts = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])

    matrix = cv2.getPerspectiveTransform(sourcePts, destPts)
    warped = cv2.warpPerspective(frame, matrix, (800, 800))
    return warped