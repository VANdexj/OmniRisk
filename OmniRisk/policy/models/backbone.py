import time
import torch
import torch.nn
from policy.models.resnet import resnet18


class _CircularConv2d(torch.nn.Conv2d):
    """Circular padding for Conv2d implemented with explicit torch.cat (convertible by torch2trt),
    instead of padding_mode='circular', whose internal F.pad(mode='circular') is unsupported by torch2trt
    and produces a wrong engine. Numerically identical to circular padding. Injected by reassigning
    __class__ on an existing Conv2d, so weight/bias names are unchanged and old checkpoints still load."""
    def _conv_forward(self, input, weight, bias):
        ph, pw = self.padding
        # positive-index slices only (torch2trt's getitem converter rejects negative)
        if pw > 0:
            W = input.shape[3]
            input = torch.cat((input[:, :, :, W - pw:W], input, input[:, :, :, 0:pw]), dim=3)
        if ph > 0:
            H = input.shape[2]
            input = torch.cat((input[:, :, H - ph:H, :], input, input[:, :, 0:ph, :]), dim=2)
        return torch.nn.functional.conv2d(input, weight, bias, self.stride,
                                          (0, 0), self.dilation, self.groups)


def _make_padding_circular(module: torch.nn.Module):
    """Panoramic 360° input: switch every padded Conv2d to circular padding so the azimuth
    borders (az=-180° / +180°) wrap seamlessly. Note: circular padding also applies to the height axis (elevation
    does not really wrap), but only on a 1~3px border, so the effect is small. Validate by training on real panoramic data.
    Uses _CircularConv2d (cat-based) instead of padding_mode='circular' so it converts to TensorRT."""
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d) and m.padding != (0, 0):
            m.__class__ = _CircularConv2d


# input: [1, 96, 384] (panoramic 360°)
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
        _make_padding_circular(self.cnn)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [1, 32, 64])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Dynamic-head encoder: mask(1)+velocity-image(3) → grid feature, same spatial
# output as the depth backbone so it concats cleanly. Kept separate from the
# shared depth backbone so the static head never sees mask/velocity.
class DynamicBackbone(torch.nn.Module):
    def __init__(self, output_dim: int, in_channels: int = 4):
        super(DynamicBackbone, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
        _make_padding_circular(self.cnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


def StaticBackbone(output_dim):
    return ResNet18(output_dim)


if __name__ == '__main__':
    net = StaticBackbone(64, 3)
    input_ = torch.zeros((1, 1, 96, 160))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
