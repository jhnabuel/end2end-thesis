"""
robot_receiver.py  —  UDP receiver that runs ON the robot (Raspberry Pi).

Receives control packets from the PC-side controller and drives the motors
via a standard dual H-bridge (L298N / L293D / TB6612) wired to GPIO.

Packet format (ASCII, sent by robot-client-pygame-joystick-udp.py):
    "<seq>,<speed>,<steering>"
    seq      : uint16, wraps at 65535 — used to detect & drop stale packets
    speed    : int, 0-100 (percentage of max throttle)
    steering : int, -50 to +50  (negative = left, positive = right)

Safety features:
    • Watchdog timer  — motors stop automatically if no packet arrives within
                        WATCHDOG_S seconds (default 0.3 s). Prevents runaway
                        if the PC crashes or Wi-Fi drops.
    • Sequence filter — out-of-order / duplicate packets are silently dropped.
    • Graceful stop   — SIGINT / KeyboardInterrupt drives motors to 0 before exit.

Motor wiring (L298N example, adjust PIN constants below):
    ENA  → PWM pin for left/forward motor speed
    IN1  → direction pin A
    IN2  → direction pin B
    ENB  → PWM pin for right/steering motor speed
    IN3  → direction pin C
    IN4  → direction pin D
    GND  → common ground with Pi
    +5V  → Pi 5 V header (logic supply; motor supply goes to L298N 12 V)

Run on the Pi:
    python robot_receiver.py
    python robot_receiver.py --port 5000 --watchdog 0.3 --pwm-freq 1000
"""

import argparse
import socket
import time
import signal
import sys

# ---------------------------------------------------------------------------
# Try to import RPi.GPIO; fall back to a stub so the code can be tested on
# a non-Pi machine (e.g. the development laptop).
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False
    print("[WARN] RPi.GPIO not found — running in SIMULATION mode (no GPIO output).")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UDP_IP       = "0.0.0.0"   # Listen on all interfaces
UDP_PORT     = 5000         # Must match PORT in robot-client-pygame-joystick-udp.py
WATCHDOG_S   = 0.3          # Seconds before auto-stop on packet loss
PWM_FREQ     = 1000         # Hz for motor PWM

# --- GPIO pin numbers (BCM numbering) ---
# Throttle / forward-backward motor (left side of L298N)
ENA = 12    # PWM-capable: GPIO 12 (Pin 32)
IN1 = 23    # GPIO 23 (Pin 16)
IN2 = 24    # GPIO 24 (Pin 18)

# Steering motor (right side of L298N)
ENB = 13    # PWM-capable: GPIO 13 (Pin 33)
IN3 = 27    # GPIO 27 (Pin 13)
IN4 = 22    # GPIO 22 (Pin 15)

# ---------------------------------------------------------------------------
# Sequence-number tracker
# ---------------------------------------------------------------------------
UINT16_MAX   = 0xFFFF
HALF_RANGE   = UINT16_MAX // 2

def is_newer_seq(incoming: int, last_seen: int) -> bool:
    """
    Robust uint16 sequence comparison that handles wrap-around.
    Returns True if `incoming` is strictly newer than `last_seen`.
    """
    if last_seen == -1:
        return True
    diff = (incoming - last_seen) & UINT16_MAX
    # diff == 0 means duplicate; diff > HALF_RANGE means it wrapped backwards
    return 0 < diff <= HALF_RANGE


# ---------------------------------------------------------------------------
# GPIO / motor helpers
# ---------------------------------------------------------------------------
class MotorController:
    """
    Thin wrapper around two H-bridge channels.

    Channel A: throttle  (speed 0-100  → forward / stop / reverse)
    Channel B: steering  (angle -50…0…+50 → left / straight / right)
    """

    def __init__(self, pwm_freq: int = PWM_FREQ):
        self._pwm_freq  = pwm_freq
        self._pwm_a     = None
        self._pwm_b     = None
        self._last_duty_a = 0
        self._last_duty_b = 0
        self._setup()

    def _setup(self):
        if not ON_PI:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        self._pwm_a = GPIO.PWM(ENA, self._pwm_freq)
        self._pwm_b = GPIO.PWM(ENB, self._pwm_freq)
        self._pwm_a.start(0)
        self._pwm_b.start(0)
        print(f"[GPIO] Motor controller initialised (PWM @ {self._pwm_freq} Hz).")

    # ------------------------------------------------------------------
    def set(self, speed: int, steering: int):
        """
        Apply throttle + steering values as received from the PC.

        speed    : 0 – 100   (0 = stop, positive = forward)
        steering : -50 – +50 (negative = left, positive = right)
        """
        speed    = int(max(-100, min(100, speed)))
        steering = int(max(-50,  min(50,  steering)))

        # ── Throttle channel ───────────────────────────────────────────
        duty_a = abs(speed)
        if speed > 0:
            fwd, rev = True, False
        elif speed < 0:
            fwd, rev = False, True
        else:
            fwd, rev = False, False

        # ── Steering channel ───────────────────────────────────────────
        duty_b  = abs(steering) * 2   # map [-50,50] → [0,100] duty
        if steering > 0:
            right, left = True, False
        elif steering < 0:
            right, left = False, True
        else:
            right, left = False, False

        if ON_PI:
            GPIO.output(IN1, GPIO.HIGH if fwd  else GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH if rev  else GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH if right else GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH if left  else GPIO.LOW)

            # Only update PWM duty if it actually changed (avoids PWM glitches)
            if duty_a != self._last_duty_a:
                self._pwm_a.ChangeDutyCycle(duty_a)
                self._last_duty_a = duty_a
            if duty_b != self._last_duty_b:
                self._pwm_b.ChangeDutyCycle(duty_b)
                self._last_duty_b = duty_b
        else:
            # Simulation: just print
            direction = "FWD" if fwd else ("REV" if rev else "STOP")
            turn      = "RIGHT" if right else ("LEFT" if left else "STR8")
            print(f"  [SIM] throttle={speed:>4}% ({direction:<4})  "
                  f"steering={steering:>4} ({turn})")

    def stop(self):
        """Full stop: zero duty, all direction pins LOW."""
        self.set(0, 0)

    def cleanup(self):
        """Release GPIO resources."""
        self.stop()
        if ON_PI:
            if self._pwm_a:
                self._pwm_a.stop()
            if self._pwm_b:
                self._pwm_b.stop()
            GPIO.cleanup()
            print("[GPIO] Cleaned up.")


# ---------------------------------------------------------------------------
# UDP receiver loop
# ---------------------------------------------------------------------------
def run(port: int, watchdog_s: float, pwm_freq: int, verbose: bool):
    motors = MotorController(pwm_freq=pwm_freq)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))
    # Non-blocking with a short timeout so the watchdog can tick even when
    # no packet arrives.
    sock.settimeout(watchdog_s)

    last_seq      = -1
    last_packet_t = time.monotonic()
    dropped_seq   = 0
    total_rx      = 0
    is_stopped    = True   # track whether we already sent a stop

    print(f"[RECV] Listening on {UDP_IP}:{port}  (watchdog={watchdog_s}s)")
    print("[RECV] Waiting for first packet …")

    # Install clean shutdown on SIGTERM (systemd / kill) in addition to SIGINT
    def _shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            # ── Receive ────────────────────────────────────────────────
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                # Watchdog: no packet within timeout → emergency stop
                elapsed = time.monotonic() - last_packet_t
                if elapsed >= watchdog_s and not is_stopped:
                    motors.stop()
                    is_stopped = True
                    print(f"[WATCHDOG] No packet for {elapsed:.2f}s — motors stopped.")
                continue

            now = time.monotonic()

            # ── Parse ──────────────────────────────────────────────────
            try:
                parts = data.decode("ascii").strip().split(",")
                if len(parts) == 3:
                    # New format: seq,speed,steering
                    seq, speed, steering = int(parts[0]), int(parts[1]), int(parts[2])
                elif len(parts) == 2:
                    # Legacy format: speed,steering  (no sequence number)
                    seq      = -2           # sentinel: always accept legacy
                    speed    = int(parts[0])
                    steering = int(parts[1])
                else:
                    raise ValueError(f"unexpected field count: {len(parts)}")
            except (ValueError, UnicodeDecodeError) as exc:
                print(f"[WARN] Malformed packet from {addr}: {data!r}  ({exc})")
                continue

            # ── Sequence filter ────────────────────────────────────────
            if seq != -2 and not is_newer_seq(seq, last_seq):
                dropped_seq += 1
                if verbose:
                    print(f"[DROP] seq={seq} (last={last_seq})  "
                          f"total_dropped={dropped_seq}")
                continue

            if seq != -2:
                last_seq = seq
            total_rx += 1
            last_packet_t = now
            is_stopped = False

            # ── Apply ──────────────────────────────────────────────────
            motors.set(speed, steering)

            if verbose:
                print(f"[RX]  seq={seq:>5}  speed={speed:>4}  "
                      f"steering={steering:>4}  from={addr[0]}")

    except KeyboardInterrupt:
        print("\n[RECV] Interrupted — stopping motors …")
    finally:
        motors.cleanup()
        sock.close()
        print(f"[RECV] Total packets received: {total_rx}  |  "
              f"Out-of-order dropped: {dropped_seq}")
        print("[RECV] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UDP motor-control receiver for the DAVE-2 robot (runs on Raspberry Pi)."
    )
    p.add_argument("--port",      type=int,   default=UDP_PORT,
                   help=f"UDP port to listen on (default: {UDP_PORT})")
    p.add_argument("--watchdog",  type=float, default=WATCHDOG_S,
                   help=f"Stop motors if no packet for this many seconds "
                        f"(default: {WATCHDOG_S})")
    p.add_argument("--pwm-freq",  type=int,   default=PWM_FREQ,
                   help=f"PWM frequency in Hz for motor speed (default: {PWM_FREQ})")
    p.add_argument("--verbose",   action="store_true",
                   help="Print every received packet (noisy at 30 Hz)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        port       = args.port,
        watchdog_s = args.watchdog,
        pwm_freq   = args.pwm_freq,
        verbose    = args.verbose,
    )
