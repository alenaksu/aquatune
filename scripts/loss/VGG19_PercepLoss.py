import torch
import torch.nn as nn
from torchvision import models


class VGG19_PercepLoss(nn.Module):
    """Calculates perceptual loss in vgg19 space."""

    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if _pretrained_ else None).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.vgg.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, image):
        image = torch.clamp(image, 0.0, 1.0)
        return (image - self.mean) / self.std

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'}
        features = {}
        x = self.preprocess(image)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer] - pred_f[layer]) ** 2)
