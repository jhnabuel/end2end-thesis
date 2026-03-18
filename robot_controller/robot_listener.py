import socket
import json

UDP_IP = "0.0.0.0"
UDP_PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for driving commands on port {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        command = json.loads(data.decode('utf-8'))

        steer = command.get("steer", 0.0)
        throttle = command.get("throttle", 0.0)

        #motor control code
        #???
        print(f"Received -> Steering : {steer} | Throttle : {throttle}")
except KeyboardInterrupt:
    print("Shutting down robot.")
