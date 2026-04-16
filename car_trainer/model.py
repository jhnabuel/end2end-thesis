import torch
import torch.nn as nn
import torch.nn.functional as F

class DAVE2(nn.Module):
    def __init__(self):
        super(DAVE2, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        self.fc_steering = nn.Linear(10, 1)
        self.fc_throttle  = nn.Linear(10, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):                 
        x = (x / 127.5) - 1.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.dropout(x)

        steering = torch.tanh(self.fc_steering(x))  # range [-1, 1]
        throttle  = torch.sigmoid(self.fc_throttle(x))  # range [0, 1]

        return steering, throttle