import pygame
import cv2
import socket
import json
import csv
import os
import time

#rasberry pi's config
PI_IP = ""
PI_PORT = ""

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#camera
cap = cv2.VideoCapture(0)

#resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No Controller Found")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected to controller: {joystick.get_name()}")

frame_count = 0
TUB_PATH = "../data/tub_01"

if not os.path.exists(TUB_PATH):
    os.makedirs(TUB_PATH, exist_ok=True)

print("Starting data collection, Press 'Q' on the video window or 'ESC' to stop")
try:
    while True:
        pygame.event.pump()
        steering = joystick.get_axis(0)
        throttle = -joystick.get_axis(1)
        if abs(steering) < 0.1: steering = 0.0

        command = {"steer": round(steering, 3), "throttle": round(throttle, 3)}
        sock.sendto(json.dumps(command).encode('utf-8')) (PI_IP, PI_PORT)

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            continue

        if throttle != 0.0 or steering != 0.0:
            image_filename = f"{frame_count}_cam-image_array_.jpg"
            
            image_filepath = os.path.join(TUB_PATH, image_filename)

            cv2.imwrite(image_filepath, frame)

            record = {
                "cam/image_array": image_filename,
                "user/angle": steering,
                "user/throttle": throttle
            }

            json_filename = f"record_{frame_count}.json"
            with open(os.path.join(TUB_PATH, json_filename), 'w') as f:
                json.dump(record, f)
            
            frame_count += 1
            
        cv2.imshow("Overhead Camera Feed", frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    stop_command = {"steer" : 0.0, "throttle" : 0.0}
    sock.sendto(json.dumps(stop_command))
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Data collection complete-- shut down safely.")