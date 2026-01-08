import torch.nn as nn
from torchvision import models


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()

        self.backbone = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Identity()
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        feats = self.backbone.features(x)
        pooled = self.backbone.avgpool(feats)
        out = self.head(pooled)
        return out

    def forward_features(self, x):
        return self.backbone.features(x)
