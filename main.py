import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.utils import GetCorrectPredCount
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.gradcam import generate_gradcam, plot_gradcam_images


from utils.utils import get_incorrrect_predictions, plot_incorrect_predictions, wrong_predictions

import matplotlib.pyplot as plt
import seaborn as sns

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# let us consider this as classes though they are not 

class_map = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from utils.dataloader import get_dataloader
from utils.utils import return_dataset_images
from model.resnet import ResNet18

train_loader, test_loader = get_dataloader()
return_dataset_images(train_loader, 12)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("CUDA Available?", use_cuda)

model = ResNet18().to(device)


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def model_summary(model, input_size):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size=input_size)

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss=0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = nn.CrossEntropyLoss()(y_pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    if scheduler!=None:
        scheduler.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))  
    return test_loss

def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


model_summary(model, (3,32,32))

# scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3,verbose=True,mode='max')
from utils.find_LR import find_lr
from utils.one_cycle_lr import get_onecycle_scheduler

optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

max_lr = find_lr(model,optimizer, criterion, device,train_loader)
if isinstance(max_lr, float):
  max_lr = max_lr
else:
  max_lr = max_lr[1]
num_epochs = 20

scheduler = get_onecycle_scheduler(optimizer,max_lr,train_loader,num_epochs)

batch_size = 512


for epoch in range(1, num_epochs+1):
  # print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion,scheduler)
  test_loss=test(model, device, test_loader)
#   scheduler.step(test_loss)

torch.save(model.state_dict(), 'model.pth')

draw_graphs()


# incorrect = get_incorrrect_predictions(model, test_loader, device)
# plot_incorrect_predictions(incorrect, class_map, count=20)

norm_mean=(0.4914, 0.4822, 0.4465) 
norm_std=(0.2023, 0.1994, 0.2010)
misclassified_images = wrong_predictions(model,test_loader, norm_mean, norm_std, class_map, device)
     

target_layers = ["layer1","layer2","layer3","layer4"]
gradcam_output, probs, predicted_classes = generate_gradcam(misclassified_images[:20], model, target_layers,device)

plot_gradcam_images(gradcam_output, target_layers, class_map, (3, 32, 32),predicted_classes, misclassified_images[:20])