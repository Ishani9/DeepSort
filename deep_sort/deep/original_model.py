import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=751, use=False):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3,2,padding=1),

            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AvgPool2d((8,4),1)
        )
        self.use = use
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        if self.use:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = Net()
    x = torch.randn(4,3,128,64)
    y = net(x)


