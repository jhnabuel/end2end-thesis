import socket
import pygame
import time
#import errno

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

# Create a UDP socket
try:
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as e:
    print(f"Error creating socket: {e}")
    exit()

#client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#client.connect((ROBOT_IP, PORT))

def draw_text(text, x, y, color=WHITE):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

#try:
#    while True:

while True:
     try:
        pygame.event.pump()
        screen.fill(BLACK) # Clear screen
        
        # Read Inputs
        raw_speed = -joystick.get_axis(1) # left stick, y axis
        raw_steer = joystick.get_axis(3)  # right stick, x axis
        is_turbo = joystick.get_button(5) # RB/R1
        is_stop = joystick.get_button(1)  # B/Circle
        
        # Calculate Values
        multiplier = 1.0 if is_turbo else 0.5
        speed = int(raw_speed * 100 * multiplier) if abs(raw_speed) > DEADZONE else 0
        steering = int(raw_steer * 50) if abs(raw_steer) > DEADZONE else 0
        
        if is_stop: speed, steering = 0, 0

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
        
        # Send Data
        #client.send(f"{speed},{steering}".encode())
        client.sendto(f"{speed},{steering}".encode(), (ROBOT_IP, PORT))
        
        time.sleep(0.05)

     except KeyboardInterrupt:
        break
     except socket.error as e:
        print(f"Socket error: {e}")
        
#except Exception as e:
#    print(f"Error: {e}")
#finally:
#    client.close()
#    pygame.quit()

client.close()
pygame.quit()
