import cv2
import numpy as np
import pickle

# Checkerboard dimensions - count INNER corners (squares-1)
# e.g. a 9x6 board of squares has 6x8 inner corners
CHECKERBOARD = (6, 8)
SQUARE_SIZE = 20  # set to real size in mm if you want real units, otherwise leave as 1.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

cap = cv2.VideoCapture(0)
collected = 0
NEEDED = 20  # collect this many good frames

print(f"Show checkerboard to camera. Need {NEEDED} captures. Press 's' to skip a bad frame, 'q' to finish early.")

while collected < NEEDED:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, found)
        cv2.putText(display, f"FOUND - captured {collected}/{NEEDED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Auto-capture with a small delay so you can move the board between shots
        objpoints.append(objp)
        imgpoints.append(corners2)
        collected += 1
        cv2.imshow("Calibration", display)
        cv2.waitKey(500)  # pause 500ms so board can be moved
    else:
        cv2.putText(display, f"No board found - {collected}/{NEEDED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Calibration", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 10:
    print(f"Only got {len(objpoints)} frames, need at least 10. Redo calibration.")
else:
    print("Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (640, 480), None, None
    )
    print(f"Reprojection error: {ret:.4f}  (good if < 0.5)")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients:\n{dist}")

    with open("calibration.pkl", "wb") as f:
        pickle.dump({"mtx": mtx, "dist": dist}, f)
    print("Saved to calibration.pkl")