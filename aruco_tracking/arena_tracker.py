import cv2
import cv2.aruco as aruco

def main():
    cap = cv2.VideoCapture(0)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Print detected marker IDs
            print("Detected IDs:", ids.flatten())

        # Show the frame
        cv2.imshow("Aruco Marker Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()