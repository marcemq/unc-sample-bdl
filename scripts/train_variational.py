import torch
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append('.')
from models.mlp_variational import MLP_variational
from utils.train_utils import train_variational, inference_variational
from utils.plot_utils import plot
from utils.data_utils import getDatasetsTrainVal
from matplotlib import pyplot as plt

BATCH_SIZE = 40
LEARNING_RATE = 1e-4
EPOCHS = 100
MC_ITS=20

def main():
    # set fixed random seed
    torch.manual_seed(42)
    
    # Check current device to work with
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get batched datasets ready to iterate 
    batched_train_data, batched_val_data = getDatasetsTrainVal(BATCH_SIZE)
    # model definition, output 2, one for mean and second for std
    model = MLP_variational(1, 100, 1).to(device)
    # model training
    train_variational(model, device, batched_train_data, batched_val_data, LEARNING_RATE, EPOCHS, mc_its=MC_ITS)
    # model inference and plot
    model.eval()
    x, y_gt, y_pred = inference_variational(model, device, batched_val_data, MC_ITS)
    plot(x, y_gt, y_pred)

if __name__ == "__main__":
    main()