############################################################################
# TopNetwork.it                                                            #
# Script: train_mauro.py                                                   #
# Date: 05/09/2025                                                         #
# Author: Mauro Chiandone + Gianluca Marini                                #
# Usage: train_mauro.py <parametri>                                        #
# Parameters:                                                              #
#   --lr", type=float, default=0.001, LEARNING RATE                        #
#   --auto_lr", type=bool, default=False, LEARNING RATE SCHEDULING(ON/OFF) # 
#   --batch_size", type=int, default=16, BATCH SIZE                        #
#   --num_classes", type=int, default=10, NUMBER OF CLASSES                #
#   --epochs", type=int, default=20, NUMBER OF EPHOCS                      #
# ##########################################################################
# This script trains a cnn model and logs tensorboard infos                #
# during training process                                                  #
############################################################################

from LeNet import LeNet
from torch.utils.tensorboard import SummaryWriter
import time, os, sys
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
import argparse

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
if __name__=="__main__":
    #################  LeNet Params  #####################
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--auto_lr", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    LEARNING_RATE = args.lr
    DYNAMIC_LR = args.auto_lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = args.num_classes
    
    
    #################  Dataset dir ####################
    WRITER_DIR = "c:\\Users\\mauro\\dev\\vehicle-classifier\\logs"
   

    ######################  TensorBoard Logger ################
    RUN_NAME = "xxx"
    writer = SummaryWriter(f'{WRITER_DIR}/{RUN_NAME}')
    
   
#    hparams = {
#        'model': f'{model._get_name()}',
#        'Loss': f'{criterion._get_name()}',
#        'optimizer': f'{optimizer.__class__}',
#        'learning_rate': LEARNING_RATE,
#        'batch_size': BATCH_SIZE,
#        'epochs': EPOCHS,
#    }
    
#    hparams = {
#        'learning_rate': LEARNING_RATE,
#        'batch_size': BATCH_SIZE,
#        'epochs': EPOCHS,
#    }
#      
#    final_metrics = {
#            'hparam/best_train_accuracy': best_train_acc,  
#            'hparam/best_val_accuracy': best_val_acc,  
#            'hparam/final_loss': train_loss         
#        }
#    

    for i in range(5):
        writer.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})



#    writer.add_hparams(hparams, final_metrics)
    
    ###########################################################
    
    
    writer.close()
