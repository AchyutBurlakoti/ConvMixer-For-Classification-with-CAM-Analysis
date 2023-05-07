import torch.nn as nn
import torch

from torchvision.models.resnet import resnet50

class Residual(nn.Module):
    
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(dim)
        
    def forward(self, y):
        x = self.conv1(y)
        x = self.gelu(x)
        x = self.batchnorm(x)
        return x + y
    
class ConvMixerBlock(nn.Module):
    
    def __init__(self, dim, kernel_size):
        super().__init__()
        
        self.resblock = Residual(dim, kernel_size)
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.gelu1 = nn.GELU()
        self.batchnorm1 = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        x = self.resblock(x)
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.batchnorm1(x)
        
        return x
    
class ConvMixer(nn.Module):
    
    def __init__(self, dim, depth, kernel_size, patch_size, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.gelu1 = nn.GELU()
        self.batchnorm1 = nn.BatchNorm2d(dim)
        
        self.convmixblock = nn.Sequential(*[ConvMixerBlock(dim, kernel_size) for i in range(depth)])
        
        self.gvp = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # self.linear = nn.Sequential(nn.Linear(dim, n_classes))
        self.linear = nn.Linear(dim, n_classes)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.gelu1(x1)
        x3 = self.batchnorm1(x2)
        
        for block in self.convmixblock:
            x3 = block(x3)
            
        x4 = self.gvp(x3)
        x5 = self.flatten(x4)
        x6 = self.linear(x5)
        
        return x6
    
class ResNetFeatureModel(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

        # Let's use pretrained resnet18 for image classification
        pretrained_resnet = resnet50(pretrained=True)

        # Extract the model layers up-to output_layer. For our case we've set output_layer = 'avg_pooling'
        self.children_list = []
        for n,c in pretrained_resnet.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(2048, 2)
         ) # 2048 cause output of resnet is this
        
    def forward(self,x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
    
def get_model(MODEL_NAME='conv-mix'):

    if MODEL_NAME == 'conv-mix':
        model = ConvMixer(dim=768, depth=32, kernel_size=7, patch_size=4)
    elif MODEL_NAME == 'res-net':
        model = ResNetFeatureModel(output_layer='avgpool')

    return model
