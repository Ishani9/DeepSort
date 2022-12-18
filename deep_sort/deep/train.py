import os
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from tqdm import tqdm
from deep_model import Net
import numpy as np

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    cudnn.benchmark = True

# data loading
root = "./dataset/"
train_dir = os.path.join(root,"train")
test_dir = os.path.join(root,"val")

transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((128,64),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def plot_graph(train_loss_curve, train_acc_curve, test_loss_curve, test_acc_curve):
    # Training Loss vs Epochs
    plt.plot(range(len(train_loss_curve)), train_loss_curve, label="Train Loss")
    plt.plot(range(len(test_loss_curve)), test_loss_curve, label="Test Loss")
    plt.title("Epoch Vs Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Loss_curves.jpg")
    plt.show()

    # Test Loss vs Epochs
    plt.plot(range(len(train_acc_curve)), train_acc_curve, label="Train Accuracy")
    plt.plot(range(len(test_acc_curve)), test_acc_curve, label="Test Accuracy")
    plt.title("Epoch Vs Accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("Accuracy_curves.jpg")
    plt.show()
    


def train_test_model():
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False)

    num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
    net = Net(num_classes=num_classes).to(device)
    
    no_epochs = 40
    best_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    train_loss_curve = []
    test_loss_curve=[]
    train_acc_curve = []
    test_acc_curve=[]

    for epoch in range(0, no_epochs):
        if epoch== 20 or epoch == 30:
            for params in optimizer.param_groups:
                params['lr'] = params['lr'] * 0.1

        print("\nEpoch : %d"%(epoch+1))
        net.train()
        training_loss = 0
        correct = 0
        total = 0
        for i, (data, labels) in enumerate(tqdm(trainloader)):
            data, labels = data.to(device),labels.to(device)
            out = net(data)
            loss = criterion(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            correct += out.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        train_loss_curve.append(training_loss/len(trainloader))
        train_acc_curve.append((correct/total)*100)
        print("Training:   Loss:", training_loss/len(trainloader), "Accuracy: ",round((correct/total)*100,3),"%")
            
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(testloader):
                data, labels = data.to(device), labels.to(device)
                out = net(data)
                loss = criterion(out, labels)

                test_loss += loss.item()
                correct += out.max(dim=1)[1].eq(labels).sum().item()
                total += labels.size(0)
        curr_acc = (correct/total)*100
        test_loss_curve.append(test_loss/len(testloader))
        test_acc_curve.append(curr_acc)
        
        print("Test    :   Loss:", test_loss/len(testloader), "Accuracy: ",round(curr_acc,3),"%")

        if best_accuracy<curr_acc:
            best_accuracy = curr_acc
            torch.save(net.state_dict(), './checkpoint/model_orginal_lr2030.pth')
    
    print("Training Finished")
    print("Best Accuracy: ", round(best_accuracy,3),"%")
    return train_loss_curve, train_acc_curve, test_loss_curve, test_acc_curve

if __name__ == '__main__':
    train_loss_curve, train_acc_curve, test_loss_curve, test_acc_curve = train_test_model()
    plot_graph(train_loss_curve, train_acc_curve, test_loss_curve, test_acc_curve)
    