import torch
import sys
sys.path.append('.')
from models.mlp_variational import MLP_variational
from utils.train_utils import train_variational, inference_variational
from utils.plot_utils import plotCompleteData, plotWithVariance
from utils.data_utils import getDatasetsTrainVal, getDatasetTestUnseen
from matplotlib import pyplot as plt

BATCH_SIZE = 40
LEARNING_RATE = 1e-3
EPOCHS = 10000
MC_ITS = 20

def main():
    # set fixed random seed
    torch.manual_seed(42)
    
    # Check current device to work with
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get batched datasets ready to iterate 
    batched_train_data, batched_val_data = getDatasetsTrainVal(BATCH_SIZE)
    batched_test_data = getDatasetTestUnseen(BATCH_SIZE)
    # model definition
    model = MLP_variational(1, 200, 1).to(device)
    # model training
    train_variational(model, device, batched_train_data, batched_val_data, LEARNING_RATE, EPOCHS, compute_pi=False, mc_its=MC_ITS)

    # model inference and plot
    model.eval()
    infr_data = inference_variational(model, device, batched_train_data, MC_ITS)
    infr_dataT = inference_variational(model, device, batched_test_data, MC_ITS)
    plotWithVariance(data=infr_data, dataT=infr_dataT, sigma_scale=1, title="Variational model")
    plt.show()

if __name__ == "__main__":
    main()