import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from orignal_model import NeuralFactory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     

PATH_LABEL = 'Dataset/data.txt'
PATH_IMAGES_FOLDER = 'Dataset/Data/'
PATH_IMAGES_FLOW_FOLDER = 'Dataset/images_flow/'

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240,320)),
    transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))
])


def plot_graph(groundtruth_curve, pred_curve, jumps):
    # Training Loss vs Epochs
    plt.plot(range(0,len(groundtruth_curve)*jumps,jumps), groundtruth_curve, label="Ground Truth")
    plt.plot(range(0,len(pred_curve)*jumps,jumps), pred_curve, label="Estimate")
    plt.title("Ground Truth vs Prediction")
    plt.legend()
    plt.xlabel("Frames")
    plt.ylabel("Speed")
    plt.savefig("Test_Result.jpg")
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
 
        curr_image = cv2.imread(fim) #/255
        
        flow_image_bgr = (flow_image_1 + flow_image_2 + flow_image_3 + flow_image_4)/4
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)        
        combined_image = 0.1*curr_image + flow_image_bgr

        x = tfms(combined_image)
        y = float(self.labels[idx+4].split()[0])
        
        return x, torch.Tensor([y])


dataset = DS(PATH_LABEL, PATH_IMAGES_FOLDER, PATH_IMAGES_FLOW_FOLDER)

full_size = len(dataset)
train_size = int(0.8 * full_size)
test_size = full_size - train_size

train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, full_size)))

# allloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
# allloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=False)
allloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)

def eval_network(model, allloader, jumps):
  gt = []
  estimate = []
  gt_sp = 0
  pd_sp = 0
  with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(allloader)):    
      images = images.to(device)
      labels = labels.to(device)
      pred = model(images.float())
      gt_sp = gt_sp + labels.item()
      pd_sp = pd_sp + pred.item()
      if idx%jumps==0:
        gt.append(gt_sp/jumps)
        estimate.append((pd_sp/jumps))
        gt_sp = 0
        pd_sp = 0

  return gt, estimate

jumps = 100
model = NeuralFactory()
model.eval().load_state_dict(torch.load('checkpoints/model_weights.pt', map_location='cpu'))
model = model.to(device)

gt, estimate = eval_network(model, allloader, jumps)
plot_graph(gt, estimate,jumps)
