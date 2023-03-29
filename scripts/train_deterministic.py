import torch
import sys
sys.path.append('.')
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.mlp_gaussian import MLP_gaussian1, MLP_gaussian2
from matplotlib import pyplot as plt
from utils.plot_utils import plot
from utils.data_utils import getDatasets
from utils.train_utils import train_deterministic, inference_deterministic

#sys.path.append('.')

# Fixed Hyperparameters
BATCH_SIZE    = 40
LEARNING_RATE = 1e-3
EPOCHS        = 10000

def main():
    # set fixed random seed
    torch.manual_seed(42)

    # Check current device to work with
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get batched datasets ready to iterate
    batched_train_data, batched_val_data = getDatasets(BATCH_SIZE)
    # model definition
    model = MLP_gaussian2(1,300,1).to(device)
    # model training
    train_deterministic(model, device, batched_train_data, batched_val_data, LEARNING_RATE, EPOCHS)

    # inference
    # ASK: wil be worth it to generate data in the range in between?
    # like data completely never spot before, or will be the same as a test batch data?
    # TODO: generate data in 1-1.5
    # TODO: transform this model in  a model to stimate aleactoric uncertanity
    # output: 2
    # loss: log verosimitud of gaussian //Kendall paper, eq2
    # ideally the stimate varianza should be the one we´ve used to generate data
    # use it ffor data in 1-1.5 range

    # model inference and plot
    model.eval()
    x, y_gt, y_pred = inference_deterministic(model, device, batched_train_data)
    plot(x, y_gt, y_pred)

if __name__ == "__main__":
    main()
