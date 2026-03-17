import pygame
import socket
HOST = ''
PORT = ''

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, world')
    data = s.recv(1024)

print(f'Received {data.decode()!r}')


pygame.init()
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("Test Input")

clock = pygame.time.Clock()
running = True
dt = 0
throttle = 0
steering = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        throttle += 0.1 * dt
    if keys[pygame.K_s]:
        throttle -= 0.1 * dt
    if keys[pygame.K_a]:
        steering += 1 * dt
    if keys[pygame.K_d]:
        steering -= 1 * dt

    pygame.display.flip()

    dt = clock.tick(60) / 1000

    print(f"throttle: {throttle:.2f}, steering: {steering:.2f}")

pygame.quit()