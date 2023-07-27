import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


def return_dataset_images(train_loader, total_images):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(total_images):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        # plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.imshow(batch_data[i].permute(1,2,0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def get_incorrrect_predictions(model, loader, device):
    """Get all incorrect predictions
    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu
    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions
    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break

def wrong_predictions(model,test_loader, norm_mean, norm_std, classes, device):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])  

            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

        plot_misclassified(wrong_predictions, norm_mean, norm_std, classes)

    return wrong_predictions
    
def plot_misclassified(wrong_predictions, norm_mean, norm_std, classes):
    fig = plt.figure(figsize=(10,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*norm_std[j])+norm_mean[j]

        img = np.transpose(img, (1, 2, 0)) #/ 2 + 0.5
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.set_title(f'\nactual : {classes[target.item()]}\npredicted : {classes[pred.item()]}',fontsize=10)
        ax.imshow(img)

    plt.show()