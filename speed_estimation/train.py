
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from orignal_model import NeuralFactory

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    cudnn.benchmark = True

PATH_LABEL = 'Dataset/data.txt'
PATH_IMAGES_FOLDER = 'Dataset/Data/'
PATH_IMAGES_FLOW_FOLDER = 'Dataset/images_flow/'

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240,320)),
    transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))
])


def plot_graph(train_loss_curve, test_loss_curve):
    # Training Loss vs Epochs
    plt.plot(range(len(train_loss_curve)), train_loss_curve, label="Train Loss")
    plt.plot(range(len(test_loss_curve)), test_loss_curve, label="Test Loss")
    plt.title("Epoch Vs Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Loss_curves.jpg")
    plt.show()


class DS(Dataset):
    def __init__(self, label_path, path_im, path_flow):
        self.labels = open(label_path).readlines()
        self.path_im = path_im
        self.path_flow = path_flow
        self.n_images = 20390
          
    def __len__(self): 
      return self.n_images

    def __getitem__(self, idx):

        f1 = self.path_flow + str(idx+1) + '.jpg'
        f2 = self.path_flow + str(idx+2) + '.jpg'
        f3 = self.path_flow + str(idx+3) + '.jpg'
        f4 = self.path_flow + str(idx+4) + '.jpg'
        fim = self.path_im +str(idx+4) + '.jpg'

        flow_image_1 = cv2.imread(f1)
        flow_image_2 = cv2.imread(f2)
        flow_image_3 = cv2.imread(f3)
        flow_image_4 = cv2.imread(f4)
        curr_image = cv2.imread(fim)

        flow_image_bgr = (flow_image_1 + flow_image_2 + flow_image_3 + flow_image_4)/4
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)        
        combined_image = 0.1*curr_image + flow_image_bgr

        x = tfms(combined_image)
        y = float(self.labels[idx+4].split()[0])
        
        return x, torch.Tensor([y])


dataset = DS(PATH_LABEL, PATH_IMAGES_FOLDER, PATH_IMAGES_FLOW_FOLDER )

full_size = len(dataset)
train_size = int(0.8 * full_size)
test_size = full_size - train_size

train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, full_size)))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2,shuffle=False)

def train_network(model, optimizer, criterion, trainloader, testloader, epochs):

  train_loss = []
  validation_loss = []

  for epoch in range(epochs):
    train_loss_sum = 0
    print("Epoch ",epoch)
    for idx, (images, labels) in enumerate(tqdm(trainloader)):
    
      model.train()
      
      images = images.to(device)
      labels = labels.to(device)
      
      outputs = model(images.float())
      
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
            
      optimizer.step()
      train_loss_sum = train_loss_sum + loss.item()
    
    train_loss.append(train_loss_sum/len(trainloader))
    print("Training Loss: ", train_loss_sum/len(trainloader))

    val_loss_sum=0
    with torch.no_grad():
      for idx, (images, labels) in enumerate(testloader):
                
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images.float())
        val_loss = criterion(outputs, labels)

        val_loss_sum = val_loss_sum + val_loss.item()

      validation_loss.append(val_loss_sum/len(testloader))
      print("Validation Loss: ", val_loss_sum/len(testloader))
    torch.save(model.state_dict(), "checkpoints/model_weights.pt")
  return train_loss, val_loss


model = NeuralFactory().to(device)
epochs = 12
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=10e-5, weight_decay=0.001)
train_loss_values, test_loss_values = train_network(model, optimizer, criterion, trainloader, testloader, epochs)
plot_graph(train_loss_values, test_loss_values)
torch.save(model.state_dict(), "checkpoints/model_weights.pt")