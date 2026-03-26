import cv2
import socket
import pygame
import time
from datetime import datetime
import json
from test_path_ren import main_path_renderer

# --- Setup ---
ROBOT_IP = '192.168.0.2'
PORT     = 5000
DEADZONE = 0.1

# Colors
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
GREEN, RED, BLUE = (0, 255, 0), (255, 0, 0), (0, 100, 255)

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Visual Window Setup
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Robot Telemetry")
font = pygame.font.SysFont("Arial", 24)

# Connect to ground robot
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((ROBOT_IP, PORT))


# Connect using UDP 
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def draw_text(text, x, y, color=WHITE):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

record_index = 0
catalog_index = 0
current_session_id = ""
last_record_time = 0
catalog_file = f"catalog_{catalog_index}".catalog
is_recording = False

robot_data = {
    'index' : '0',
    'session_id' : '',
    'timestamp_ms' : 0,
    'cam/image_array': "",
    'angle' : 0.0,
    'user/mode' : "user",
    'throttle' : 0.0
}

camera_generator = main_path_renderer()

try:
    with open(catalog_file, 'a') as f:
        while True:
            frame = next(camera_generator, None)

            pygame.event.pump()
            screen.fill(BLACK) # Clear screen
            
            # Read Inputs
            raw_speed = -joystick.get_axis(1) # left stick, y axis
            raw_steer = joystick.get_axis(3)  # right stick, x axis
            is_turbo = joystick.get_button(5) # RB/R1
            is_stop = joystick.get_button(1)  # B/Circle

            #start/stop recording logic
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 2:
                        is_recording = not is_recording
                    
                    if is_recording:
                        print("Recording Started")
                    else:
                        print("Recording Stopped.")
                

            # Calculate Values 
            multiplier = 1.0 if is_turbo else 0.5
            speed = int(raw_speed * 100 * multiplier) if abs(raw_speed) > DEADZONE else 0
            steering = int(raw_steer * 50) if abs(raw_steer) > DEADZONE else 0
            
            if is_stop: speed, steering = 0, 0

            #data logging (index, session_id, timestamp_ms, cam/image_array, angle, user_mode, throttle)
            current_time = time.time()
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            image_filename = f"{current_date_str}_{record_index}.jpg"
            if is_recording and (current_time - last_record_time) >= 0.05:
                
                if frame is not None:
                    cv2.imwrite(image_filename, frame)

                robot_data['index'] = record_index
                robot_data['session_id'] = current_date_str
                robot_data['timestamp_ms'] = int(time.time()) * 1000
                robot_data['cam/image_array'] = image_filename
                robot_data['angle'] = steering
                robot_data['user/mode'] = "user"
                robot_data['throttle'] = speed

                json_string = json.dumps(robot_data)
                f.write(json_string + "\n")
                f.flush()

                last_record_time = current_time
                record_index += 1


            # --- Visual Elements ---
            # 1. Status Text
            draw_text(f"Speed: {speed}%", 20, 20)
            draw_text(f"Steer: {steering}", 20, 50)
            
            # 2. Button Indicators
            turbo_color = GREEN if is_turbo else (50, 50, 50)
            stop_color = RED if is_stop else (50, 50, 50)
            pygame.draw.rect(screen, turbo_color, (20, 100, 100, 40))
            draw_text("Turbo", 35, 108, BLACK if is_turbo else WHITE)
            
            pygame.draw.rect(screen, stop_color, (140, 100, 100, 40))
            draw_text("Stop", 160, 108, BLACK if is_stop else WHITE)

            # 3. Speed Bar
            pygame.draw.rect(screen, WHITE, (320, 50, 30, 200), 2) # Outline
            bar_height = int(abs(speed) * 2)
            bar_y = 150 - bar_height if speed >= 0 else 150
            pygame.draw.rect(screen, BLUE, (322, bar_y, 26, bar_height))

            pygame.display.flip() # Update display
            
            
            # # Send Data
            # try:
            #     udp_socket.sendto(f"{speed},{steering}".encode(), (ROBOT_IP, PORT))
            # except Exception as net_error:
            #     print(f"Send error: {net_error}")
            
            time.sleep(0.05)

except Exception as e:
    print(f"Error: {e}")
finally:
    udp_socket.close()
    pygame.quit()
