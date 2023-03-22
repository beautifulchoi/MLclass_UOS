from torch import nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms

class ResNet50(nn.Module):
    def __init__(self,in_channels=1, num_classes=10):
        super(ResNet50, self).__init__()
        cnn=models.resnet50(pretrained=False)
        self.cnn=cnn
        self.num_classes=num_classes

        #Change input channel 3->1
        self.cnn.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Change the output layer to output 10 classes instead of 1000 classes
        fc_in = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(fc_in, 10)

    def forward(self, x):
        return self.cnn(x)


class Base(nn.Module):
#https://wikidocs.net/63565
    def __init__(self):
        super(Base, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)
    
        # 전결합층 한정으로 가중치 초기화
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out