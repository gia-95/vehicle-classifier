from LeNet import LeNet
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import transforms


##################  TensorBoard Param ################
RUN_NAME = f'leNet_run_{int(time.time()*1000)}'

#################  LeNet Param  #####################
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 16

#################  Dataset dir ####################
TRAIN_DATASET_DIR = '/Users/gmarini/dev/vehicle-classifier/dvc/data/train'
VAL_DATASET_DIR = '/Users/gmarini/dev/vehicle-classifier/dvc/data/val'


################    TRAIN      ###########################

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = 'cpu'

# writer = SummaryWriter(f'../logs/{RUN_NAME}')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor()
])

model = LeNet()

trainable_parameters = [ p for p in model.parameters() if p.requires_grad ] 

criterion = MSELoss()

optimizer = torch.optim.Adam(trainable_parameters, lr = LEARNING_RATE )


train_dataset = ImageFolder(TRAIN_DATASET_DIR, transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = ImageFolder(VAL_DATASET_DIR, transform)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

model = model.to(device)

for epoch in range(EPOCHS) :    
    
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader :
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        
        _, preds = torch.max(outputs, 1)
        
        breakpoint()
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    train_loss = running_loss / len(train_dataset)
    
    train_acc = running_corrects.double() / len(train_dataset)

