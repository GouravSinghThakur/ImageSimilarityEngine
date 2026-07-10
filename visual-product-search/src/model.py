import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    Resnet50 Feature Extractor 
    Convert an Iput image to 2048 embedding vectors
    """
    def __init__(self):
        super().__init__()
        self.model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc=nn.Identity()
    def forward(self,x):
        return self.model(x)
    
