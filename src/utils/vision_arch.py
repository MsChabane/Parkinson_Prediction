import sys
import pickle
import torch
import torch.nn as nn
from torchvision import models


class ConfigurableVisionClassifier(nn.Module):
    def __init__(
        self,
        base_model_name: str = "mobilenet_v3_large",
        pretrained: bool = False,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
        freeze_ratio: float = 0.5,
    ):
        super().__init__()
        self.base_model_name = base_model_name.lower()
        self.backbone, in_features = self._build_backbone(self.base_model_name, pretrained)
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def _build_backbone(self, name: str, pretrained: bool):
        if name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            return nn.Sequential(net.features, net.avgpool), net.classifier[0].in_features
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            return nn.Sequential(net.features, net.avgpool), net.classifier[0].in_features
        elif name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            return nn.Sequential(net.features, nn.AdaptiveAvgPool2d((1, 1))), net.classifier[1].in_features
        elif name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            return nn.Sequential(*list(net.children())[:-1]), net.fc.in_features
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            return nn.Sequential(*list(net.children())[:-1]), net.fc.in_features
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            return nn.Sequential(net.features, net.avgpool), net.classifier[1].in_features
        elif name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = models.densenet121(weights=weights)
            return nn.Sequential(net.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))), net.classifier.in_features
        raise ValueError(f"Unsupported model: {name}")

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_embedding(x)).squeeze(-1)


def safe_load_pickle(path):
    setattr(sys.modules["__main__"], "ConfigurableVisionClassifier", ConfigurableVisionClassifier)
    with open(path, "rb") as f:
        return pickle.load(f)