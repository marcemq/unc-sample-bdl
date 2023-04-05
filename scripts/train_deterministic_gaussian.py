import torch
import sys
sys.path.append('.')
from models.mlp_gaussian import MLP_gaussian
from matplotlib import pyplot as plt
from utils.plot_utils import plotWithVariance
from utils.data_utils import getDatasetsTrainVal, getDatasetTestUnseen
from utils.train_utils import train_deterministic_gaussian, inference_deterministic_gaussian

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
    batched_train_data, batched_val_data = getDatasetsTrainVal(BATCH_SIZE, dataFileName="dataS03.csv")
    batched_test_data = getDatasetTestUnseen(BATCH_SIZE, dataFileName="dataUnseenS03.csv")
    # model definition
    model = MLP_gaussian(1,300,1).to(device)
    # model training
    train_deterministic_gaussian(model, device, batched_train_data, batched_val_data, LEARNING_RATE, EPOCHS)

    # model inference and plot
    model.eval()
    infr_data = inference_deterministic_gaussian(model, device, batched_train_data)
    infr_dataT = inference_deterministic_gaussian(model, device, batched_test_data)
    plotWithVariance(data=infr_data, dataT=infr_dataT, sigma_scale=1, title="Deterministic Gaussian model")
    plt.show()

if __name__ == "__main__":
    main()
