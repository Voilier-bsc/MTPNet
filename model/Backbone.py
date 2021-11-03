import torch
from torch import nn
from torchvision.models import mobilenet_v2
from typing import Tuple


def trim_network_at_index(network: nn.Module, index: int = -1) :
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])

def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]):
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]

class MobileNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = trim_network_at_index(mobilenet_v2(pretrained=True), -1)

        # input     : [batch_size, n_channels, length, width]
        # output    : [batch_size, n_convolution_filters(1280)]
    def forward(self, input_tensor: torch.Tensor):
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])                       #global everage pooling

