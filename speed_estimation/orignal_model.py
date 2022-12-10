import torch.nn as nn

class NeuralFactory(nn.Module):
  def __init__(self):
    super(NeuralFactory, self).__init__()
    
    self.all_conv_layers = nn.Sequential(
        # Convolution 1
        nn.Conv2d(3, 24, kernel_size=(5,5), stride=2), #, padding=1),
        # nn.BatchNorm2d(24),
        nn.ELU(),
        
        # Convolution 2
        nn.Conv2d(24, 36, kernel_size=(5,5), stride=2), #, padding=1),
        nn.BatchNorm2d(36),
        nn.ELU(),

      
        # Convolution 3
        nn.Conv2d(36, 48, kernel_size=(5,5), stride=2), #, padding=1),
        nn.BatchNorm2d(48),
        nn.ELU(),
        nn.Dropout(0.5),
      
        # Convolution 4
        nn.Conv2d(48, 64, kernel_size=(5,5), stride=2), #, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),

        # Convolution 5
        nn.Conv2d(64, 64, kernel_size=(4,4), stride=2), #, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU())

    self.layers2 = nn.Sequential(nn.Linear(in_features=2240, out_features=100), nn.ELU(), 
                                 nn.Linear(100, 50), nn.ELU(), 
                                 nn.Linear(50, 10), nn.ELU(), 
                                 nn.Linear(10, 1))

  def forward(self, x):
    x = self.all_conv_layers(x)
    y = x.view(x.size(0), -1)
    x = self.layers2(y)
    return x

if __name__ == '__main__':
    net = NeuralFactory()