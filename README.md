# Image classification - CIFAR add.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/), 
[PyTorch](https://pytorch.org/), 
[torchvision](https://github.com/pytorch/vision) 0.8, 
Uses [matplotlib](https://matplotlib.org/)  for ploting accuracy and losses.

## Info

 * we are training our custom resnet model on CIFAR dataset. 
 * we are using a custom convolutional neural network (CNN) architectures which includes skip connections etc
 * Implemented in pytorch 
 * in this repo, i am exporing pytorch-lightning. by using this model that we train i am going to publish space app.
 * this repo only does training using pytorch-lightning


## About

* transforms.py contains the transforms we used using Albumentation library
* dataset.py has dataset class for applying transforms 
* dataloader.py contains dataloaders which downloads and gives dataloaders. 
* find_LR.py contains a function which fetches max_lr we can use for onecyclelr
* one_cycle_lr returns us one_cycle scheduler
* model.py has pytorch settings 
* utils.py has some graph functions and others 
* notebook has all pytorch_lightning results 

## Results 


### Test accuracy 

* 0.9194999933242798 

## Usage

```bash
git clone https://github.com/srikanthp1/S12.git
```
* notebook for pytorch_lightning


## Model details

```python
model = model().to(device)
summary(model, input_size=(3, 32, 32))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 32, 32]          73,728
        GroupNorm-14          [-1, 128, 32, 32]             256
           Conv2d-15          [-1, 128, 32, 32]         147,456
        GroupNorm-16          [-1, 128, 32, 32]             256
           Conv2d-17          [-1, 128, 32, 32]           8,192
        GroupNorm-18          [-1, 128, 32, 32]             256
       BasicBlock-19          [-1, 128, 32, 32]               0
           Conv2d-20          [-1, 128, 32, 32]         147,456
        GroupNorm-21          [-1, 128, 32, 32]             256
           Conv2d-22          [-1, 128, 32, 32]         147,456
        GroupNorm-23          [-1, 128, 32, 32]             256
       BasicBlock-24          [-1, 128, 32, 32]               0
           Conv2d-25          [-1, 256, 16, 16]         294,912
        GroupNorm-26          [-1, 256, 16, 16]             512
           Conv2d-27          [-1, 256, 16, 16]         589,824
        GroupNorm-28          [-1, 256, 16, 16]             512
           Conv2d-29          [-1, 256, 16, 16]          32,768
        GroupNorm-30          [-1, 256, 16, 16]             512
       BasicBlock-31          [-1, 256, 16, 16]               0
           Conv2d-32          [-1, 256, 16, 16]         589,824
        GroupNorm-33          [-1, 256, 16, 16]             512
           Conv2d-34          [-1, 256, 16, 16]         589,824
        GroupNorm-35          [-1, 256, 16, 16]             512
       BasicBlock-36          [-1, 256, 16, 16]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 27.00
Params size (MB): 42.63
Estimated Total Size (MB): 69.64
----------------------------------------------------------------

```

## Analysis 

* setting was done right as model class can have all extra attributes 
* accuracy drop is not observed 
* validation accuracy reached more than 90 percentage 
* oncecycle is set to end at 50 
* will use this model to share transformers app in space 

* loss graphs

![alt text](https://github.com/srikanthp1/S12/blob/master/images/loss_acc_graphs.png)

* misclassified 

![alt text](https://github.com/srikanthp1/S12/blob/master/images/misclassified_images.png)

* grad cam

![alt text](https://github.com/srikanthp1/S12/blob/master/images/grad_cam.png)

* lr change 

![alt text](https://github.com/srikanthp1/S12/blob/master/images/lr_change.png)

