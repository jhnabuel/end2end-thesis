import socket
import RPi.GPIO as GPIO
#import errno

UDP_IP      = '0.0.0.0'
UDP_PORT    = 5000
BUFFER_SIZE = 1024

# Define BCM Pins for L298N
#ENA, IN1, IN2 = 18, 23, 24  # Left Motor
#ENB, IN3, IN4 = 13, 27, 22  # Right Motor

ENA, IN1, IN2 = 25, 23, 24
ENB, IN3, IN4 = 17, 22, 27

GPIO.setmode(GPIO.BCM)
for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)

# Setup PWM for speed control
pwm_left  = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(0)
pwm_right.start(0)

def set_motors(speed, steering):
    # Basic Differential Logic
    left_target = speed + steering
    right_target = speed - steering

    for s, p, i1, i2 in [(left_target, pwm_left, IN1, IN2), 
                         (right_target, pwm_right, IN3, IN4)]:
        # Set speed (0-100)
        p.ChangeDutyCycle(min(abs(s), 100))
        # Set direction
        GPIO.output(i1, GPIO.HIGH if s >= 0 else GPIO.LOW)
        GPIO.output(i2, GPIO.LOW if s >= 0 else GPIO.HIGH)

# Create a UDP Socket
try:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((UDP_IP, UDP_PORT))
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

except socket.error as e:
    print(f"Error binding socket: {e}")
    exit()

while True:
    try:
        response, _ = server.recvfrom(BUFFER_SIZE)
        if response:
            data = response.decode()
            throttle, steering = map(int, data.split(','))
            set_motors(throttle, steering)
    except KeyboardInterrupt:
        break
    except socket.error as e:
        print(f"Socket error: {e}")

# loop terminates, do some cleanup
pwm_left.stop()
pwm_right.stop()
GPIO.cleanup()
server.close()

# Socket Server Setup
#server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server.bind(('0.0.0.0', 5000))
#server.listen(1)
#print("UDP Server: Robot ready. Waiting for connection...")

#try:
    #conn, addr = server.accept()
#    while True:
        #data = conn.recv(1024).decode()
#        response, _ = server.recvfrom(1024)
#        if not response: break
        # Command format: "speed, steering"
#        data = response.decode()
#        speed, steering = map(int, data.split(','))
#        set_motors(speed, steering)

#except BrokenPipeError as e:
#    print(f"Caught BrokenPipeError: {e}. Client disconnected.")
#    pwm_left.stop()
#    pwm_right.stop()
#    GPIO.cleanup()
#    server.close()

#except socket.error as e:
#    if e.errno == errno.EPIPE:
#        print(f"Caught socket.error EPIPE: {e}. Client disconnected.")
#    else: raise

#finally:
#    pwm_left.stop()
#    pwm_right.stop()
#    GPIO.cleanup()
#    server.close()
