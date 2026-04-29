import torch
import torch.nn as nn
import torch.nn.functional as F

class DAVE2(nn.Module):
    def __init__(self):
        super(DAVE2, self).__init__()
        
        # Convolutional layers with BatchNorm
        # bias=False because BatchNorm has its own learnable bias term
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False)
        self.bn2   = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False)
        self.bn3   = nn.BatchNorm2d(48)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.bn5   = nn.BatchNorm2d(64)

        # Fully connected layers with BatchNorm
        # 64 * 1 * 18 = 1152 verified for input size 66x200
        self.fc1    = nn.Linear(64 * 1 * 18, 100)
        self.bn_fc1 = nn.BatchNorm1d(100)
        
        self.fc2    = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        
        self.fc3    = nn.Linear(50, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)

        # Single steering output — throttle is fixed at inference time
        self.fc_steering = nn.Linear(10, 1)

        # Reduced dropout to 0.1 — BatchNorm already provides regularization
        # High dropout (0.5) combined with BatchNorm causes variance shift
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Normalize pixels from [0, 255] to [-1, 1]
        x = (x / 127.5) - 1.0

        # Conv block: Conv → BatchNorm → Activation
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.elu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)

        # FC block: Linear → BatchNorm → Activation → Dropout
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc3(self.fc3(x)))

        # Steering output bounded to [-1, 1]
        steering = torch.tanh(self.fc_steering(x))
        return steering