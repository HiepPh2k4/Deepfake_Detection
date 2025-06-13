import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)  # (batch, 256, H, W)
        x = F.avg_pool2d(x, (x.size(2), 1))  # (batch, 256, 1, W)
        x = x.squeeze(2)  # (batch, 256, W)
        x = x.permute(0, 2, 1)  # (batch, W, 256)
        x, _ = self.lstm(x)  # (batch, W, 256)
        x = x[:, -1, :]  # (batch, 256)
        x = self.fc(x)
        return x