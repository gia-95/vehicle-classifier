#################################################################
# TopNetwork.it                                                 #
#################################################################
# Script: LeNet.py                                              #
# Date: 04/09/2025                                              #
# Author: Gianluca Marini + Mauro Chiandone                     #
# Usage: 
#   from LeNet import LeNet
#   myLeNet = LeNet(number_of_classes)
# ###############################################################
# This script implements a LeNet CNN class from scratch         # 
# uses PyTorch framework                                       #
#################################################################

import torch
from torch import nn
import torch.nn.functional as F


class LeNetFeatExtractor(nn.Module) :
    def __init__ (self) :
        super(LeNetFeatExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 16, 3)
    def forward(self, x) :
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x
        
class LeNetClassifier(nn.Module):
    def __init__(self,NUMBER_OF_CLASSES):
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUMBER_OF_CLASSES)
    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet(nn.Module):
    def __init__(self,NUMBER_OF_CLASSES):
        super(LeNet, self).__init__()
        self.feat = LeNetFeatExtractor()
        self.classifer = LeNetClassifier(NUMBER_OF_CLASSES)
    def forward(self, x):
        x = self.feat(x)
        x = self.classifer(x)
        return x
 
