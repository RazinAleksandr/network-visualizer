import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input channel (grayscale), 16 output channels, 3x3 conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 14x14
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 7x7
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class ResNetMNIST(nn.Module):
    def __init__(self):
        super(ResNetMNIST, self).__init__()
        # Load standard ResNet18
        self.resnet = resnet18(weights=None)
        # Modify input layer for 1 channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify output layer for 10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

class AlexNetMNIST(nn.Module):
    def __init__(self):
        super(AlexNetMNIST, self).__init__()
        # AlexNet usually expects 224x224. We can either:
        # 1. Upsample input (SLOW)
        # 2. Modify architecture to handle 28x28 (Better for efficiency)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 7x7
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Standard SRCNN: 9-1-5
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        # No activation at output (reconstructing pixel values)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Contracting path
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(32, 64, 3, padding=1)
        
        # Expansive path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Conv2d(64 + 32, 32, 3, padding=1) # Skip connection from enc2
        self.dec1 = nn.Conv2d(32 + 16, 16, 3, padding=1) # Skip connection from enc1
        
        self.out = nn.Conv2d(16, 2, 1) # 2 classes: bg, fg

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(self.pool(x1)))
        
        # Bottleneck
        b = torch.relu(self.bottleneck(self.pool(x2)))
        
        # Decoder with skip connections
        d2 = self.up(b)
        # Resize if needed (simplification: assume power of 2 match)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = torch.relu(self.dec2(d2))
        
        d1 = self.up(d2)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = torch.relu(self.dec1(d1))
        
        return self.out(d1)

class SimpleDiffUNet(nn.Module):
    def __init__(self):
        super(SimpleDiffUNet, self).__init__()
        # Contracting path
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 128, 3, padding=1)
        
        # Expansive path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Conv2d(128 + 64, 64, 3, padding=1) 
        self.dec1 = nn.Conv2d(64 + 32, 32, 3, padding=1) 
        
        self.out = nn.Conv2d(32, 1, 1) # Predicts NOISE (1 channel)

    def forward(self, x, t):
        # Time embedding
        # t: (batch_size, 1)
        t_emb = self.time_embed(t.float()) # (B, 64)
        t_emb = t_emb.unsqueeze(2).unsqueeze(3) # (B, 64, 1, 1)
        
        # Encoder
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(self.pool(x1))) # (B, 64, 7, 7)
        
        # Inject time embedding to x2
        # We broadcast add it
        x2 = x2 + t_emb.expand(-1, -1, x2.shape[2], x2.shape[3]) # Simple adding conditioning
        
        # Bottleneck
        b = torch.relu(self.bottleneck(self.pool(x2))) # (B, 128, 3, 3)
        
        # Decoder
        d2 = self.up(b)
        # Pad if needed (Simplification for 28x28 MNIST: 3->6 vs 7. Need explicit padding or better sizing)
        # MNIST is 28x28. 
        # Pool1: 14x14
        # Pool2: 7x7
        # Pool3 (Bottleneck input): 3x3 ? No pool 3 in simple version.
        # Let's check dims:
        # Input: 28x28
        # enc1: 28x28
        # pool1: 14x14 -> x1 (Wait, enc1 is before pool) -> No, x1 is enc1 output (28x28)
        # pool(x1) -> 14x14
        # enc2: 14x14 -> x2 
        # pool(x2) -> 7x7
        # bottleneck: 7x7 -> b
        # up(b) -> 14x14. Correct.
        
        d2 = torch.cat((d2, x2), dim=1) # 128 (b upsampled? no b is 128) -> b is 128, up sends to ... wait.
        # up does not change channels. b is 128. up(b) is 128.
        # x2 is 64. 128+64 = 192.
        # dec2 input needs 192.
        # My init says 128+64 -> 64. Correct.
        d2 = torch.relu(self.dec2(d2)) # 64
        
        d1 = self.up(d2) # 28x28
        d1 = torch.cat((d1, x1), dim=1) # 64 + 32 = 96
        # dec1 input needs 96.
        # My init says 64+32. Correct.
        d1 = torch.relu(self.dec1(d1))
        
        return self.out(d1)

def get_model(name):
    if name == "SimpleCNN":
        return SimpleCNN()
    elif name == "ResNet18":
        return ResNetMNIST()
    elif name == "AlexNet":
        return AlexNetMNIST()
    elif name == "SRCNN":
        return SRCNN()
    elif name == "SimpleUNet":
        return SimpleUNet()
    elif name == "SimpleDiffUNet":
        return SimpleDiffUNet()
    else:
        raise ValueError(f"Unknown model {name}")
