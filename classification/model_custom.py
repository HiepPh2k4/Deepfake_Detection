import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Entry Flow (6 layers)
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Approximated SeparableConv2D
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Strided Conv
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Residual Block Definition
        def residual_block(in_channels, out_channels, stride=1):
            return nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            ])

        # Downsample Block Definition
        def downsample_block(in_channels, out_channels):
            return nn.ModuleList([
                # Shortcut
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
                # Main Path
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels)
            ])

        # Middle Flow
        # Stage 1: 5 Residual Blocks (64 filters, 20 layers)
        self.stage1 = nn.ModuleList([residual_block(64, 64) for _ in range(5)])

        # Stage 2: 1 Downsample + 3 Residual Blocks (128 filters, 14 layers)
        self.stage2_downsample = downsample_block(64, 128)
        self.stage2 = nn.ModuleList([residual_block(128, 128) for _ in range(3)])

        # Stage 3: 1 Downsample + 2 Residual Blocks (256 filters, 14 layers)
        self.stage3_downsample = downsample_block(128, 256)
        self.stage3 = nn.ModuleList([residual_block(256, 256) for _ in range(2)])

        # Exit Flow (4 layers)
        self.exit_flow = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.entry_flow(x)

        # Stage 1
        for block in self.stage1:
            shortcut = x
            x = block[0](x)
            x = block[1](x)
            x = block[2](x)
            x = block[3](x)
            x = block[4](x)
            x = x + shortcut
            x = nn.functional.relu(x)

        # Stage 2
        shortcut = self.stage2_downsample[0](x)
        shortcut = self.stage2_downsample[1](shortcut)
        x = self.stage2_downsample[2](x)
        x = self.stage2_downsample[3](x)
        x = x + shortcut
        x = nn.functional.relu(x)
        for block in self.stage2:
            shortcut = x
            x = block[0](x)
            x = block[1](x)
            x = block[2](x)
            x = block[3](x)
            x = block[4](x)
            x = x + shortcut
            x = nn.functional.relu(x)

        # Stage 3
        shortcut = self.stage3_downsample[0](x)
        shortcut = self.stage3_downsample[1](shortcut)
        x = self.stage3_downsample[2](x)
        x = self.stage3_downsample[3](x)
        x = x + shortcut
        x = nn.functional.relu(x)
        for block in self.stage3:
            shortcut = x
            x = block[0](x)
            x = block[1](x)
            x = block[2](x)
            x = block[3](x)
            x = block[4](x)
            x = x + shortcut
            x = nn.functional.relu(x)

        x = self.exit_flow(x)
        return x