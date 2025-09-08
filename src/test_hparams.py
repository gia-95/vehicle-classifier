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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

#########################################################################################################
if __name__=="__main__":
    
    
    #################  Dataset dir ####################
    WRITER_DIR = "c:\\Users\\mauro\\dev\\vehicle-classifier\\logs"
   

    ######################  TensorBoard Logger ################
    RUN_NAME = "xxx"
    writer = SummaryWriter(f'{WRITER_DIR}/{RUN_NAME}')
    def simulate_train_model(hparams_dict):
        hidden_dim=hparams_dict["hidden_dim"]
        lr=hparams_dict["lr"]
        acc = lr*100
        loss = hidden_dim/64
        return {"accuracy": acc, "loss": loss}
     

    # Logging con TensorBoard
    def log_experiment(hparams_dict):
        metrics = simulate_train_model(hparams_dict)
        writer.add_hparams(hparams_dict, metrics)

    # Esempi di run con diversi hparams
    experiments = [
        {"lr": 0.01, "hidden_dim": 64, "epochs": 5},
        {"lr": 0.001, "hidden_dim": 128, "epochs": 5},
        {"lr": 0.0001, "hidden_dim": 256, "epochs": 5},
    ]

    for exp in experiments:
        log_experiment(exp)   
   
    writer.close()
