import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleStrongCNN(nn.Module):
    """Simple but strong CNN for deepfake detection - based on proven architectures"""

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(SimpleStrongCNN, self).__init__()

        # Feature extraction backbone (similar to VGG but optimized)
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 5: 14x14 -> 7x7
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


class ResNetBasedCNN(nn.Module):
    """ResNet-based CNN for deepfake detection using pretrained backbone"""

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ResNetBasedCNN, self).__init__()

        # Use ResNet50 as backbone
        from torchvision import models
        resnet = models.resnet50(pretrained=True)

        # Remove the final fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze early layers (optional)
        for i, child in enumerate(self.backbone.children()):
            if i < 6:  # Freeze first 6 layers
                for param in child.parameters():
                    param.requires_grad = False

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


class EfficientNetBasedCNN(nn.Module):
    """EfficientNet-based CNN for deepfake detection"""

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(EfficientNetBasedCNN, self).__init__()

        try:
            from torchvision import models
            # Use EfficientNet-B0 as backbone
            efficientnet = models.efficientnet_b0(pretrained=True)

            # Remove classifier
            self.backbone = efficientnet.features

            # Get the number of features
            num_features = efficientnet.classifier[1].in_features

            # Custom classifier
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Dropout(dropout_rate * 0.7),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, num_classes)
            )

        except:
            # Fallback to simple model if EfficientNet not available
            print("EfficientNet not available, using SimpleStrongCNN")
            self.backbone = None
            self.simple_model = SimpleStrongCNN(num_classes, dropout_rate)

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
            x = self.classifier(x)
        else:
            x = self.simple_model(x)
        return x


# Backward compatibility
class HybridDeepfakeCNN(nn.Module):
    """Use SimpleStrongCNN as HybridDeepfakeCNN for backward compatibility"""

    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(HybridDeepfakeCNN, self).__init__()
        # Use the proven simple strong model
        self.model = SimpleStrongCNN(num_classes=num_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.model(x)


class ImprovedDeepfakeCNN(nn.Module):
    """Use ResNetBasedCNN as ImprovedDeepfakeCNN"""

    def __init__(self, num_classes=1, width_mult=1.0, depth_mult=1.0):
        super(ImprovedDeepfakeCNN, self).__init__()
        # Use ResNet-based model for better performance
        self.model = ResNetBasedCNN(num_classes=num_classes, dropout_rate=0.5)

    def forward(self, x):
        return self.model(x)


# Loss functions
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_models():
    """Test model outputs and parameters"""
    print("Testing model architectures...")

    # Test input
    x = torch.randn(2, 3, 224, 224)

    # Test SimpleStrongCNN
    model1 = SimpleStrongCNN(num_classes=1)
    output1 = model1(x)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"SimpleStrongCNN: Output {output1.shape}, Params: {params1 / 1e6:.2f}M")

    # Test ResNetBasedCNN
    try:
        model2 = ResNetBasedCNN(num_classes=1)
        output2 = model2(x)
        params2 = sum(p.numel() for p in model2.parameters())
        trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        print(
            f"ResNetBasedCNN: Output {output2.shape}, Params: {params2 / 1e6:.2f}M, Trainable: {trainable2 / 1e6:.2f}M")
    except Exception as e:
        print(f"ResNetBasedCNN failed: {e}")

    # Test backward compatibility
    model3 = HybridDeepfakeCNN(num_classes=1)
    output3 = model3(x)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"HybridDeepfakeCNN (compatibility): Output {output3.shape}, Params: {params3 / 1e6:.2f}M")


if __name__ == "__main__":
    test_models()