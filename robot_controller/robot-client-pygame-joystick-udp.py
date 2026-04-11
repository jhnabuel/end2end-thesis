import sys
import os

# Allow importing from car_trainer/ regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAR_TRAINER = os.path.join(_ROOT, "car_trainer")
if _CAR_TRAINER not in sys.path:
    sys.path.insert(0, _CAR_TRAINER)

import cv2
import socket
import pygame
import time
from datetime import datetime
import json
import threading
from queue import Queue, Empty

from test_path_renderer import main_path_renderer
from inference import InferenceEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROBOT_IP          = '192.168.0.2'
PORT              = 5000
DEADZONE          = 0.1
FRAME_INTERVAL    = 0.05
CATALOG_FILE      = "../data/catalog_0.catalog"
IMAGE_DIR         = "../data/images"
DELETE_COUNT      = 60
WEIGHTS_PATH      = os.path.join(_CAR_TRAINER, "dave2_robot_model.pth")

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
GREEN  = (0,   255, 0)
RED    = (255, 0,   0)
BLUE   = (0,   100, 255)
YELLOW = (255, 220, 0)
PURPLE = (160, 60,  255)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
state_lock  = threading.Lock()
shared_state = {
    'speed':        0,
    'steering':     0,
    'is_turbo':     False,
    'is_stop':      False,
    'is_recording': False,
    'is_ai_mode':   False,
}

frame_queue  = Queue(maxsize=2)
record_queue = Queue(maxsize=10)
stop_event   = threading.Event()
catalog_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def load_catalog_entries():
    entries = []
    if not os.path.exists(CATALOG_FILE):
        return entries
    with open(CATALOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def get_next_record_index():
    """Resume index from the last entry so we never collide across sessions."""
    entries = load_catalog_entries()
    if not entries:
        return 0
    return max(e.get('index', -1) for e in entries) + 1


def delete_last_n_frames(n, record_index_ref):
    """
    Delete the last *n* catalog entries and their image files.
    Updates record_index_ref[0] to the new next-free index.
    Returns the number of entries deleted.
    """
    with catalog_lock:
        entries = load_catalog_entries()
        if not entries:
            return 0

        to_delete = entries[-n:]
        remaining = entries[:-n]

        deleted_imgs = 0
        for entry in to_delete:
            img_path = entry.get('cam/image_array', '')
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    deleted_imgs += 1
                except OSError as e:
                    print(f"[WARN] Could not delete {img_path}: {e}")

        with open(CATALOG_FILE, 'w') as f:
            for entry in remaining:
                f.write(json.dumps(entry) + "\n")

        new_index = (remaining[-1]['index'] + 1) if remaining else 0
        record_index_ref[0] = new_index

        print(f"[DELETE] Removed {len(to_delete)} entries, "
              f"{deleted_imgs} image files.  Next index: {new_index}")
        return len(to_delete)


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def camera_thread():
    camera_generator = main_path_renderer()
    while not stop_event.is_set():
        result = next(camera_generator, None)
        if result is None:
            break
        frame, save_frame = result
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put((frame, save_frame))
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
            speed    = shared_state['speed']
            steering = shared_state['steering']

        try:
            client.sendto(f"{speed},{steering}".encode(), (ROBOT_IP, PORT))
        except socket.error as e:
            print(f"Send error: {e}")

    client.close()


def disk_writer_thread():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    while not stop_event.is_set() or not record_queue.empty():
        try:
            item = record_queue.get(timeout=0.1)
        except Empty:
            continue

        filename, frame, robot_data = item
        if frame is None:
            continue
        cv2.imwrite(filename, frame)
        with catalog_lock:
            with open(CATALOG_FILE, 'a') as f:
                f.write(json.dumps(robot_data) + "\n")
                f.flush()


def main():
    current_paddle = 0
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    screen = pygame.display.set_mode((420, 340))
    pygame.display.set_caption("Robot Telemetry")
    font       = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)

    def draw_text(text, x, y, color=WHITE, small=False):
        img = (small_font if small else font).render(text, True, color)
        screen.blit(img, (x, y))

    # Resume index from existing catalog
    record_index_ref = [get_next_record_index()]
    print(f"[INIT] Starting record index at {record_index_ref[0]}")

    # Load inference engine
    print("[INIT] Loading inference engine …")
    try:
        engine       = InferenceEngine(weights_path=WEIGHTS_PATH)
        ai_available = True
    except Exception as e:
        print(f"[WARN] Could not load inference engine: {e}")
        engine       = None
        ai_available = False

    # Feedback state
    delete_flash_until = 0.0
    delete_flash_msg   = ""

    # Safe defaults so display never crashes before first loop tick
    speed    = 0
    steering = 0
    is_ai_mode   = False
    is_recording = False
    is_stop      = False

    threads = [
        threading.Thread(target=camera_thread,      daemon=True),
        threading.Thread(target=control_thread,     daemon=True),
        threading.Thread(target=disk_writer_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    next_frame_time = time.time()

    try:
        while not stop_event.is_set():
            # --- Rate limiter ---
            now       = time.time()
            sleep_for = next_frame_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_frame_time = max(next_frame_time + FRAME_INTERVAL, time.time())

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()

                if event.type == pygame.JOYBUTTONDOWN:
                    # Button 2 → toggle recording
                    if event.button == 2:
                        with state_lock:
                            shared_state['is_recording'] = not shared_state['is_recording']
                            mode = shared_state['is_recording']
                        print("Recording Started." if mode else "Recording Stopped.")

                    # Button 3 → delete last 60 frames
                    elif event.button == 3:
                        drained = 0
                        while not record_queue.empty():
                            try:
                                record_queue.get_nowait()
                                drained += 1
                            except Empty:
                                break
                        if drained:
                            print(f"[DELETE] Drained {drained} pending frames from queue.")
                        n = delete_last_n_frames(DELETE_COUNT, record_index_ref)
                        delete_flash_msg   = f"Deleted {n} frames!"
                        delete_flash_until = time.time() + 2.0

                    # Button 0 → toggle AI mode
                    elif event.button == 0:
                        if not ai_available:
                            print("[AI] Model not loaded — AI mode unavailable.")
                        else:
                            with state_lock:
                                shared_state['is_ai_mode'] = not shared_state['is_ai_mode']
                                mode = shared_state['is_ai_mode']
                            print(f"[AI] AI mode {'ENABLED' if mode else 'DISABLED'}.")

                    # Button 4 → gear down
                    elif event.button == 4:
                        current_paddle = max(current_paddle - 1, 0)
                        print(f"[GEAR] Paddle: {current_paddle}")

                    # Button 5 → gear up
                    elif event.button == 5:
                        current_paddle = min(current_paddle + 1, 3)
                        print(f"[GEAR] Paddle: {current_paddle}")

            paddle_speed = {0: 0, 1: 0.45, 2: 0.65, 3: 0.85}

            raw_steer = joystick.get_axis(3)   # right stick X
            is_stop   = joystick.get_button(1) # B / Circle

            with state_lock:
                is_ai_mode   = shared_state['is_ai_mode']
                is_recording = shared_state['is_recording']

            try:
                frame, save_frame = frame_queue.get_nowait()
            except Empty:
                frame, save_frame = None, None

            if is_ai_mode and engine is not None and frame is not None:
                ai_steering, ai_throttle = engine.predict_frame(frame)
                speed    = 0 if is_stop else ai_throttle
                steering = 0 if is_stop else ai_steering
            else:
                raw_speed = paddle_speed[current_paddle]
                speed    = int(raw_speed * 100 * 0.5) if raw_speed > DEADZONE else 0
                steering = int(raw_steer * 50)         if abs(raw_steer) > DEADZONE else 0
                if is_stop:
                    speed, steering = 0, 0

            with state_lock:
                shared_state['speed']    = speed
                shared_state['steering'] = steering
                shared_state['is_stop']  = is_stop

            # --- Recording ---
            if is_recording and frame is not None and save_frame is not None:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
                idx              = record_index_ref[0]
                image_filename   = os.path.join(IMAGE_DIR, f"{current_date_str}_{idx}.jpg")
                robot_data = {
                    'index':           idx,
                    'session_id':      current_date_str,
                    'timestamp_ms':    int(time.time() * 1000),
                    'cam/image_array': image_filename,
                    'angle':           steering,
                    'user/mode':       'ai' if is_ai_mode else 'user',
                    'throttle':        speed,
                }
                if not record_queue.full():
                    record_queue.put((image_filename, save_frame, robot_data))
                record_index_ref[0] += 1

            # ----------------------------------------------------------------
            # Display
            # ----------------------------------------------------------------
            screen.fill(BLACK)

            if is_ai_mode:
                pygame.draw.rect(screen, PURPLE, (0, 0, 420, 32))
                draw_text("  \u2605 AI AUTOPILOT MODE \u2605", 10, 5, WHITE)

            y = 36 if is_ai_mode else 10
            draw_text(f"Speed:  {speed}%",    20, y)
            draw_text(f"Steer:  {steering}",  20, y + 30)
            draw_text(f"Frame#: {record_index_ref[0]}", 20, y + 58,  YELLOW, small=True)
            draw_text(f"Recording: {'ON' if is_recording else 'OFF'}", 20, y + 78,
                      GREEN if is_recording else WHITE)
            draw_text(f"Gear: {current_paddle}", 20, y + 98)

            stop_color = RED if is_stop else (50, 50, 50)
            pygame.draw.rect(screen, stop_color, (140, y + 112, 100, 36))
            draw_text("Stop", 160, y + 120, BLACK if is_stop else WHITE)

            bar_height = int(abs(speed) * 2)
            bar_y      = 190 - bar_height if speed >= 0 else 190
            pygame.draw.rect(screen, WHITE,  (370, 50, 30, 200), 2)
            pygame.draw.rect(screen, PURPLE if is_ai_mode else BLUE, (372, bar_y, 26, bar_height))

            draw_text("Btn2=Rec  Btn3=Del60  Btn0=AI", 10, 306, (120, 120, 120), small=True)

            if time.time() < delete_flash_until:
                draw_text(delete_flash_msg, 20, 282, RED)

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
