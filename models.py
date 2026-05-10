import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet50Face(nn.Module):
    """
    ResNet50-based face recognition backbone.

    Input:
        [B, 3, 112, 112]

    Output:
        L2-normalized embedding [B, embedding_size]
    """

    def __init__(self, embedding_size: int = 512, dropout: float = 0.4):
        super().__init__()

        backbone = resnet50(weights=None)

        # Remove avgpool and fc.
        self.body = nn.Sequential(*list(backbone.children())[:-2])

        self.head = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        self._init_head()

    def _init_head(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def build_model(backbone: str = "resnet50", embedding_size: int = 512, dropout: float = 0.4) -> nn.Module:
    backbone = backbone.lower()
    if backbone != "resnet50":
        raise ValueError(f"Only resnet50 is implemented, got: {backbone}")
    return ResNet50Face(embedding_size=embedding_size, dropout=dropout)
