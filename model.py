import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImprovedDeepfakeDetector(nn.Module):
    """
    Improved model using ResNet18 as backbone with transfer learning.
    This provides much better feature extraction for deepfake detection.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ImprovedDeepfakeDetector, self).__init__()
        # Use ResNet18 as backbone (pre-trained on ImageNet)
        # Handle both old and new PyTorch versions
        try:
            # New API (PyTorch 0.13+)
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(weights=None)
        except (AttributeError, TypeError):
            # Old API (PyTorch < 0.13)
            resnet = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers, fine-tune later layers
        for param in list(resnet.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.backbone = resnet
        
    def forward(self, x):
        return self.backbone(x)

class SimpleCNN(nn.Module):
    """
    Legacy simple CNN - kept for backward compatibility.
    For better accuracy, use ImprovedDeepfakeDetector.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # net.0
            nn.BatchNorm2d(32),
            nn.ReLU(),                                              # net.1
            nn.MaxPool2d(2, 2),                                     # net.2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # net.3
            nn.BatchNorm2d(64),
            nn.ReLU(),                                              # net.4
            nn.MaxPool2d(2, 2),                                     # net.5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Additional layer
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),                                           # net.6
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 256),                          # net.7
            nn.ReLU(),                                              # net.8
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)                                       # net.9
        )

    def forward(self, x):
        return self.net(x)
