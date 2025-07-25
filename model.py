import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(in_features=128*12*12, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=6)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=128)

    def forward(self,x):
        #(B, 3, 200, 200) -> (B, 16, 200, 200)
        x = self.conv1(x)
        x= self.bn1(x)
        x = F.relu(x)
        # (B, 16, 200, 200) -> (B, 16, 100, 100)
        x = F.max_pool2d(x,2)
        # (B, 16, 100, 100) -> (B, 32, 100, 100)
        x = self.conv2(x)
        x= self.bn2(x)
        x = F.relu(x)
        # (B, 32, 100, 100) -> (B, 32, 50, 50)
        x = F.max_pool2d(x,2)
        # (B, 32, 50, 50) -> (B, 64, 50,50)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # (B, 64, 50, 50) -> (B, 64, 25, 25)
        x = F.max_pool2d(x, 2)
        # (B, 64, 25, 25) -> (B, 128, 25, 25)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # (B, 128, 25, 25) -> (B, 128, 12, 12)
        x = F.max_pool2d(x, 2)
        # (B, 128, 12, 12) -> (B, 128*12*12)
        x = x.view(-1,128*12*12)
        # (B, 128*12*12) -> (B, 64)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        # (B, 64) -> (B, 6)
        x = self.linear2(x)
        return x
        
