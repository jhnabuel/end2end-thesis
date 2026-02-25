import cv2
import numpy as np
import cv2.aruco as aruco

def main():
    # Initialize the webcam (or use a video file path)
    cap = cv2.VideoCapture(0)

    # 1. Setup the ArUco Dictionary and Parameters
    # Change DICT_4X4_50 to match whatever dictionary your marker is from
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    # Create the detector (OpenCV 4.7+ API)
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # How far ahead the road should project (in pixels)
    LOOKAHEAD_DISTANCE = 200 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Optionally draw the basic bounding box around the marker
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                # Extract the 4 corners of the current marker
                # c[0]=Top-Left, c[1]=Top-Right, c[2]=Bottom-Right, c[3]=Bottom-Left
                c = corners[i][0] 

                # 3. Calculate Center Point
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                center_pt = (center_x, center_y)

                # 4. Calculate Front Midpoint (Assuming top edge is "forward")
                front_x = int((c[0][0] + c[1][0]) / 2)
                front_y = int((c[0][1] + c[1][1]) / 2)

                # 5. Calculate Heading Vector
                dx = front_x - center_x
                dy = front_y - center_y

                # Calculate the magnitude (length) of the vector
                magnitude = np.hypot(dx, dy)

                if magnitude > 0:
                    # Normalize the vector (make its length 1)
                    norm_dx = dx / magnitude
                    norm_dy = dy / magnitude

                    # 6. Project the Lookahead Line
                    lookahead_x = int(center_x + (norm_dx * LOOKAHEAD_DISTANCE))
                    lookahead_y = int(center_y + (norm_dy * LOOKAHEAD_DISTANCE))
                    lookahead_pt = (lookahead_x, lookahead_y)

                    # Draw the main lookahead line (The "Road")
                    cv2.line(frame, center_pt, lookahead_pt, (0, 255, 0), 4)
                    
                    # Draw a circle at the end of the lookahead line
                    cv2.circle(frame, lookahead_pt, 8, (0, 0, 255), -1)

        # Display the output
        cv2.imshow("Robot Lookahead Vision", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()