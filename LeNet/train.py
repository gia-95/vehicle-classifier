from LeNet import LeNet
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
import torch


RUN_NAME = f'leNet_run_{int(time.time()*1000)}'

writer = SummaryWriter(f'../logs/{RUN_NAME}')

model = LeNet()

sample_input = torch.randn(1, 1, 28, 28)

# Aggiungi il grafo del modello a TensorBoard
writer.add_graph(model, sample_input)

# Chiudi il writer
writer.close()