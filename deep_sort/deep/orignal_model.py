import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ci, co, s):
        super(ResidualBlock,self).__init__() 
        self.resblock = nn.Sequential(
                        nn.Conv2d(ci,co,3,stride=s,padding=1),
                        nn.BatchNorm2d(co),
                        nn.ReLU(True))

        self.prep = nn.Sequential(
                    nn.Conv2d(ci,co,3,stride=s,padding=1),
                    nn.BatchNorm2d(co))

    def forward(self,x):
        residual = x
        x = self.resblock(x)
        residual = self.prep(residual)
        x = x + residual
        return F.relu(x,True)


class Net(nn.Module):
    def __init__(self, num_classes=751, use=False):
        super(Net,self).__init__()
        self.init_layers= nn.Sequential(
            #Convolutional 1
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            #Convolutional 2
            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Maxpool 3
            nn.MaxPool2d(3,2,padding=1)           
            )

        self.resblock4 = ResidualBlock(32,32, 1)
        self.resblock5 = ResidualBlock(32,32, 1)

        self.resblock6 = ResidualBlock(32,64, 2)
        self.resblock7 = ResidualBlock(64,64, 1)

        self.resblock8 = ResidualBlock(64,128, 2)
        self.resblock9 = ResidualBlock(128,128, 1)

        self.dense10 =  nn.Sequential(
                        #Dense 10
                        nn.Linear(16384, 1028),
                        nn.BatchNorm1d(1028),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(1028, num_classes)    
                        )

        self.use = use
    
    def forward(self, x):
        x = self.init_layers(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = x.view(x.size(0),-1)

        if self.use:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x

        x = self.dense10(x)
        return x


if __name__ == '__main__':
    net = Net()
    x = torch.randn(4,3,128,64)
    y = net(x)




import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ci, co, s):
        super(ResidualBlock,self).__init__() 
        self.resblock = nn.Sequential(
                        nn.Conv2d(ci,co,3,stride=s,padding=1),
                        nn.BatchNorm2d(co),
                        nn.ReLU(True))

        self.prep = nn.Sequential(
                    nn.Conv2d(ci,co,3,stride=s,padding=1),
                    nn.BatchNorm2d(co))

    def forward(self,x):
        residual = x
        x = self.resblock(x)
        residual = self.prep(residual)
        x = x + residual
        return F.relu(x,True)


class Net(nn.Module):
    def __init__(self, num_classes=751, use=False):
        super(Net,self).__init__()
        self.init_layers= nn.Sequential(
            #Convolutional 1
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            #Convolutional 2
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Maxpool 3
            nn.AvgPool2d(3,2,padding=1)           
            )

        self.resblock4 = ResidualBlock(64,64, 1)
        self.resblock5 = ResidualBlock(64,64, 1)

        self.resblock6 = ResidualBlock(64,128, 2)
        self.resblock7 = ResidualBlock(128,128, 1)

        self.resblock8 = ResidualBlock(128,256, 2)
        self.resblock9 = ResidualBlock(256,256, 1)
        
        self.avgpool = nn.AvgPool2d((8,4),1)
        
        self.dense10 =  nn.Sequential(
                        #Dense 10
                        nn.Linear(16384, 1028),
                        nn.BatchNorm1d(1028),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(1028, num_classes)    
                        )

        self.use = use
    
    def forward(self, x):
        x = self.init_layers(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = x.view(x.size(0),-1)

        if self.use:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x

        x = self.dense10(x)
        return x


if __name__ == '__main__':
    net = Net()
    x = torch.randn(4,3,128,64)
    y = net(x)


