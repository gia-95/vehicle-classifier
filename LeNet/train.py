from LeNet import LeNet
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
########################################### INFO ########################################################
#Optimizers:
#
#   Adadelta
#   Adafactor
#   Adagrad
#   Adam
#   AdamW
#   SparseAdam
#   Adamax
#   ASGD
#   LBFGS
#   NAdam
#   RAdam
#   RMSprop
#   Rprop
#   SGD
#
#Tune the learning rate during optimization:
#
#   lr_scheduler.LRScheduler			        Adjusts the learning rate during optimization:
#   lr_scheduler.LambdaLR				        Sets the initial learning rate.
#   lr_scheduler.MultiplicativeLR			    Multiply the learning rate of each parameter group by the factor given in the specified function.
#   lr_scheduler.StepLR				            Decays the learning rate of each parameter group by gamma every step_size epochs.
#   lr_scheduler.MultiStepLR			        Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
#   lr_scheduler.ConstantLR				        Multiply the learning rate of each parameter group by a small constant factor.
#   lr_scheduler.LinearLR				        Decays the learning rate of each parameter group by linearly changing small multiplicative factor.
#   lr_scheduler.ExponentialLR			        Decays the learning rate of each parameter group by gamma every epoch.
#   lr_scheduler.PolynomialLR			        Decays the learning rate of each parameter group using a polynomial function in the given total_iters.
#   lr_scheduler.CosineAnnealingLR			    Set the learning rate of each parameter group using a cosine annealing schedule.
#   lr_scheduler.ChainedScheduler			    Chains a list of learning rate schedulers.
#   lr_scheduler.SequentialLR			        Contains a list of schedulers expected to be called sequentially during the optimization process.
#   lr_scheduler.ReduceLROnPlateau			    Reduce learning rate when a metric has stopped improving.
#   lr_scheduler.CyclicLR				        Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
#   lr_scheduler.OneCycleLR				        Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
#   lr_scheduler.CosineAnnealingWarmRestarts	Set the learning rate of each parameter group using a cosine annealing schedule.
#
#########################################################################################################

#################  LeNet Param  #####################
LEARNING_RATE = 0.001
DYNAMIC_LR = False
EPOCHS = 1
BATCH_SIZE = 16

#################  Dataset dir ####################
TRAIN_DATASET_DIR = '/Users/gmarini/dev/vehicle-classifier/dvc/data/train'
VAL_DATASET_DIR = '/Users/gmarini/dev/vehicle-classifier/dvc/data/val'


##################  TensorBoard Param ################
hparams = {
    'model': 'LeNet',
    'optimizer': 'SGD',
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
}
RUN_NAME = f"run{int(time.time()*1000)}_lr_{'DYNAMIC'if DYNAMIC_LR else hparams['learning_rate']}_bs_{hparams['batch_size']}"
writer = SummaryWriter(f'/Users/gmarini/dev/vehicle-classifier/logs/{RUN_NAME}')
writer.add_text('Train dataset', 'TRAIN_DATASET_DIR')
writer.add_text('Val dataset', 'VAL_DATASET_DIR')


################    TRAIN      ###########################

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = 'cpu'


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor()
])

train_dataset = ImageFolder(TRAIN_DATASET_DIR, transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = ImageFolder(VAL_DATASET_DIR, transform)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

############ MODEL ##############

model = LeNet()
writer.add_graph(model, torch.randn(1, 1, 32, 32))
trainable_parameters = [ p for p in model.parameters() if p.requires_grad ] 
# criterion = MSELoss()
criterion = CrossEntropyLoss(reduction='mean') # 'mean' calcolacola il valore medio della loss sul batch (maggiore stabilità numerica rispetto ad utilizzare 'sum' che somma i singoli contributi di ogni immagine del batch)
                                               #  (mettendo 'sum', nel calcolo della running_loss (loss totale dell'epoca) quindi non devi poi moltiplicare per la dimensione del batch)
optimizer = torch.optim.SGD(trainable_parameters, lr = LEARNING_RATE, momentum=0.9 )
# optimizer = torch.optim.Adam(trainable_parameters, lr = LEARNING_RATE )
if DYNAMIC_LR :
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max')
model = model.to(device)

#################################

best_train_acc = 0.0   
best_val_acc = 0.0   
final_loss = 0.0

total_labels = []
total_preds = []

for epoch in range(EPOCHS) : 
    
    current_lr = optimizer.param_groups[0]['lr']
    
    start_time = time.time() 
    
    #####  Train  #####
    
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader :
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() # Azzera il valore dei GRADIENTI per questo batch

        outputs = model(inputs)
        
        _, preds = torch.max(outputs, 1)
        
        total_labels = total_labels + labels.data.cpu().tolist()
        total_preds = total_preds + preds.data.cpu().tolist()
        
        loss = criterion(outputs, labels) # Calcola il valore dell'errore tramite la loss
        
        loss.backward() # Calcola la direzione del gradiente per i vari parametri
        
        optimizer.step() # Aggiorna pesi : SDG es. nuovo_peso = vecchio_peso - learning_rate * gradiente
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


        
    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)
    
    
    #####  Validation  #####
    
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad() :# con questa impostazione evitiamo che venga inutilmente aggiornato il grafo computazionale
    
        for inputs, labels in val_loader :
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
        
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)
        
    
    ### Aggiungi valori a TensorBoard
    if float(train_acc.item()) > best_train_acc :
        best_train_acc = float(train_acc.item())
    if float(val_acc.item()) > best_val_acc :
        best_val_acc = float(val_acc.item())
    
    writer.add_scalars("Accuracy", { "Train": train_acc, "Validation": val_acc},  epoch)
    writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
    writer.add_scalar("Learning Rate", current_lr, epoch)
    
  
        
    ### Calcolo tempo di elaborazione dell'epoca
    end_time = time.time()
    elapsed_time = end_time-start_time  
                # Ore                           # Minuti                              # Secondi
    time_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
    
   
    print(f'Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr} Train_Loss: {train_loss:.4f} Train_Acc: {train_acc:.4f} Val_Loss: {val_loss:.4f} Val_Acc: {val_acc:.4f} Elapsed_time: {time_str}')



writer.add_pr_curve('pr_curve', total_labels, total_preds)
 
final_metrics = {
        'hparam/best_train_accuracy': best_train_acc,  
        'hparam/best_val_accuracy': best_val_acc,  
        'hparam/final_loss': final_loss         
    }
writer.add_hparams(hparams, final_metrics)
writer.close()
