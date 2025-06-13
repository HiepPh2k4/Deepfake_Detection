import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn import init

class ChannelAttention(nn.Module):
    """Simple but effective channel attention module."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)

class TransformerAttention(nn.Module):
    """Transformer-based attention for spatial relationships."""
    def __init__(self, channels, num_heads=8):
        super(TransformerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (h*w, b, c)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # (b, c, h, w)
        return self.sigmoid(x)

class DualAttentionModule(nn.Module):
    """Combined channel and transformer-based spatial attention."""
    def __init__(self, channels, reduction=16, num_heads=8):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = TransformerAttention(channels, num_heads)

    def forward(self, x):
        x = self.channel_attention(x)
        att_map = self.spatial_attention(x)
        return x * att_map

class OptimizedXceptionNet(nn.Module):
    """Optimized Xception for deepfake detection with attention mechanisms."""
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(OptimizedXceptionNet, self).__init__()
        self.base_model = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        self.feat_channels = [728, 2048]  # Reduced from [256, 728, 2048]
        self.attention_modules = nn.ModuleList([
            DualAttentionModule(ch, reduction=16, num_heads=8) for ch in self.feat_channels
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(sum(self.feat_channels), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base_model(x)
        selected_features = features[-2:]  # Use last two feature maps
        attended_features = [att_module(feat) for feat, att_module in zip(selected_features, self.attention_modules)]
        pooled_features = [self.global_pool(feat).flatten(1) for feat in attended_features]
        combined = torch.cat(pooled_features, dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

    def get_attention_maps(self, x):
        """Extract spatial attention maps for visualization."""
        features = self.base_model(x)
        selected_features = features[-2:]  # Match forward
        attention_maps = []
        for feat, att_module in zip(selected_features, self.attention_modules):
            att_map = att_module.spatial_attention(feat)
            attention_maps.append(att_map)
        return attention_maps

def create_optimized_model(dropout_rate=0.5):
    """Create optimized model."""
    return OptimizedXceptionNet(num_classes=1, dropout_rate=dropout_rate)