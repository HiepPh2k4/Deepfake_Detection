import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, num_classes, d_model=192, nhead=4, num_layers=5, dropout_rate=0.3):
        super(Transformer, self).__init__()

        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Projection layer (adjusted for 299x299 input)
        self.input_projection = nn.Linear(32 * 37, d_model)  # 32 channels * 37 freq after pooling

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Additional projection to increase parameters
        self.pre_classifier = nn.Linear(d_model, d_model)

        # CTC output layer
        self.classifier = nn.Linear(d_model, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch_size, 1, 299, 299)
        features = self.conv_layers(x)  # (batch_size, 32, 37, 37)
        batch_size, channels, time, freq = features.size()
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(batch_size, time, channels * freq)  # (batch_size, 37, 32*37)
        features = self.input_projection(features)  # (batch_size, 37, d_model)
        features = self.transformer_encoder(features)  # (batch_size, 37, d_model)
        features = self.pre_classifier(features)  # (batch_size, 37, d_model)
        output = self.classifier(features)  # (batch_size, 37, num_classes)
        return output