import cv2
import socket
import pygame
import time
from datetime import datetime
import json
import threading
from queue import Queue, Empty
from test_path_renderer import main_path_renderer

# --- Setup ---
ROBOT_IP = '192.168.0.2'
PORT = 5000
DEADZONE = 0.1
FRAME_INTERVAL = 0.05

BLACK, WHITE = (0, 0, 0), (255, 255, 255)
GREEN, RED, BLUE = (0, 255, 0), (255, 0, 0), (0, 100, 255)

# --- Shared State ---
state_lock = threading.Lock()
shared_state = {
    'speed': 0,
    'steering': 0,
    'is_turbo': False,
    'is_stop': False,
    'is_recording': False,
}

frame_queue = Queue(maxsize=2)
record_queue = Queue(maxsize=10)
stop_event = threading.Event()


def camera_thread():
    camera_generator = main_path_renderer()
    while not stop_event.is_set():
        frame = next(camera_generator, None)
        if frame is None:
            break
        # Drop old frame if consumer is slow
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put(frame)
    stop_event.set()


def control_thread():
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as e:
        print(f"Socket error: {e}")
        stop_event.set()
        return

    next_send_time = time.time()
    while not stop_event.is_set():
        now = time.time()
        sleep_for = next_send_time - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_send_time = max(next_send_time + FRAME_INTERVAL, time.time())

        with state_lock:
            speed = shared_state['speed']
            steering = shared_state['steering']

        try:
            client.sendto(f"{speed},{steering}".encode(), (ROBOT_IP, PORT))
        except socket.error as e:
            print(f"Send error: {e}")

    client.close()


def disk_writer_thread():
    catalog_file = "../data/catalog_0.catalog"
    with open(catalog_file, 'a') as f:
        while not stop_event.is_set() or not record_queue.empty():
            try:
                item = record_queue.get(timeout=0.1)
            except Empty:
                continue

            filename, frame, robot_data = item
            cv2.imwrite(filename, frame)
            f.write(json.dumps(robot_data) + "\n")
            f.flush()


def main():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Robot Telemetry")
    font = pygame.font.SysFont("Arial", 24)

    def draw_text(text, x, y, color=WHITE):
        img = font.render(text, True, color)
        screen.blit(img, (x, y))

    record_index = 0

    threads = [
        threading.Thread(target=camera_thread, daemon=True),
        threading.Thread(target=control_thread, daemon=True),
        threading.Thread(target=disk_writer_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    next_frame_time = time.time()

    try:
        while not stop_event.is_set():
            # --- Rate limiter ---
            now = time.time()
            sleep_for = next_frame_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_frame_time = max(
                next_frame_time + FRAME_INTERVAL, time.time())

            # --- Input ---
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 2:
                        with state_lock:
                            shared_state['is_recording'] = not shared_state['is_recording']
                        with state_lock:
                            print(
                                "Recording Started." if shared_state['is_recording'] else "Recording Stopped.")

            raw_speed = -joystick.get_axis(1)
            raw_steer = joystick.get_axis(3)
            is_turbo = joystick.get_button(5)
            is_stop = joystick.get_button(1)

            multiplier = 1.0 if is_turbo else 0.5
            speed = int(raw_speed * 100 *
                        multiplier) if abs(raw_speed) > DEADZONE else 0
            steering = int(raw_steer * 100) if abs(raw_steer) > DEADZONE else 0
            if is_stop:
                speed, steering = 0, 0

            with state_lock:
                shared_state['speed'] = speed
                shared_state['steering'] = steering
                shared_state['is_turbo'] = is_turbo
                shared_state['is_stop'] = is_stop
                is_recording = shared_state['is_recording']

            # --- Recording ---
            try:
                frame = frame_queue.get_nowait()
            except Empty:
                frame = None

            if is_recording and frame is not None:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
                image_filename = f"../data/images/{current_date_str}_{int(time.time() * 1000)}.jpg"
                robot_data = {
                    'index': record_index,
                    'session_id': current_date_str,
                    'timestamp_ms': int(time.time() * 1000),
                    'cam/image_array': image_filename,
                    'angle': steering,
                    'user/mode': 'user',
                    'throttle': speed,
                }
                if not record_queue.full():
                    record_queue.put((image_filename, frame, robot_data))
                record_index += 1

            # --- Display ---
            screen.fill(BLACK)
            draw_text(f"Speed: {speed}%", 20, 20)
            draw_text(f"Steer: {steering}", 20, 50)
            draw_text(f"Recording: {'ON' if is_recording else 'OFF'}", 20, 80,
                      GREEN if is_recording else WHITE)

            turbo_color = GREEN if is_turbo else (50, 50, 50)
            stop_color = RED if is_stop else (50, 50, 50)
            pygame.draw.rect(screen, turbo_color, (20, 100, 100, 40))
            draw_text("Turbo", 35, 108, BLACK if is_turbo else WHITE)
            pygame.draw.rect(screen, stop_color, (140, 100, 100, 40))
            draw_text("Stop", 160, 108, BLACK if is_stop else WHITE)

            bar_height = int(abs(speed) * 2)
            bar_y = 150 - bar_height if speed >= 0 else 150
            pygame.draw.rect(screen, WHITE, (320, 50, 30, 200), 2)
            pygame.draw.rect(screen, BLUE, (322, bar_y, 26, bar_height))

            pygame.display.flip()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        pygame.quit()


if __name__ == "__main__":
    main()
