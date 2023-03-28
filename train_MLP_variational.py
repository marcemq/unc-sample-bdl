import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.mlp_variational import MLP_variational
from matplotlib import pyplot as plt

# Fixed Hyperparameters
BATCH_SIZE = 40
LEARNING_RATE = 1e-4
EPOCHS = 100
MC_ITS=20

# TODO: this function should return train, val and test batchs
# TODO: debug error due to BATCH_SIZE, test with 50 to see the reported error
def getDatasets():
    data = np.genfromtxt("data.csv", dtype=float, delimiter=',', names=True) 
    x_tensor = torch.from_numpy(data["x"]).float()
    y_tensor = torch.from_numpy(data["y"]).float()

    dataset = TensorDataset(x_tensor, y_tensor)
    train_data_size = int(len(x_tensor) * 0.8)
    val_data_size = len(x_tensor) - train_data_size
    train_data, val_data = random_split(dataset, [train_data_size, val_data_size])

    # Form batches
    batched_train_data = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    batched_val_data = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    return batched_train_data, batched_val_data

def plot(x, y_gt, y_pred):
    plt.scatter(x, y_gt, marker ='x', color='blue', s=10, label='y_GT' )
    plt.scatter(x, y_pred, marker ='x', color='green', s=10, label='y_pred' )
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $')
    plt.show()

def main():
    # set fixed random seed
    torch.manual_seed(42)
    
    # Check current device to work with
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get batched datasets ready to iterate 
    batched_train_data, batched_val_data = getDatasets()
    # model definition, output 2, one for mean and second for std
    model = MLP_variational(1,100,1).to(device)
    # model training
    train_variational(model, device, batched_train_data, batched_val_data, mc_its=MC_ITS)
    # model inference and plot
    model.eval()
    x, y_gt, y_pred = inference(model, device, batched_val_data)
    plot(x, y_gt, y_pred)

def inference(model, device, test_data):
    x, y_gt, y_pred = [], [], []
    for batch_idx, (xtest, ytest_gt) in enumerate(test_data):
        pred_sum = 0
        for mc_run in range(MC_ITS):
            xtest = xtest.to(device).reshape((len(xtest),-1))
            ytest_gt = ytest_gt.reshape((len(ytest_gt),-1))
            pred, kl = model.predict(xtest)
            pred_sum += pred
        x.append(xtest.numpy())
        y_gt.append(ytest_gt.numpy())
        y_pred.append((pred_sum/MC_ITS).numpy())
        # AKS should I comput the sdt fro the prediction and reporte it bacK?
    return x, y_gt, y_pred 

def train_variational(model, device, train_data, val_data, mc_its=10):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    list_loss_train = []
    list_loss_val = []

    # Training
    for epoch in range(EPOCHS):
        error = 0
        M = len(train_data)
        model.train()
        for batch_idx, (x, y_gt) in enumerate(train_data):
            model.zero_grad()
            
            x = x.to(device).reshape((len(x),-1))
            y_gt = y_gt.to(device).reshape((len(y_gt),-1))

            # forward pass and losses compute
            pred, nll_loss, kl_loss = model(x, y_gt, mc_its)

            # Compute loss and error
            loss = kl_loss/M + nll_loss
            error += loss.detach().item()

            # compute gradients and update weights
            loss.backward()
            optimizer.step()
        list_loss_train.append(error/M)

        #validation
        error_val = 0
        M_val = len(val_data)
        model.eval()
        with torch.no_grad():
            for batch_idx, (xval, yval_gt) in enumerate(val_data):
                xval = xval.to(device).reshape((len(xval),-1))
                yval_gt = yval_gt.to(device).reshape((len(yval_gt),-1))
                pred_val, nll_loss, kl_loss = model(x, y_gt, mc_its)
                pi     = (2.0**(M-batch_idx))/(2.0**M-1)
                loss = pi*kl_loss + nll_loss
                error_val += loss.detach().item()
        list_loss_val.append(error/M_val)

    plot_losses(list_loss_train, list_loss_val)

def plot_losses(list_loss_train, list_loss_val):
    plt.figure(figsize=(12,12))
    plt.plot(list_loss_train, label="loss train")
    plt.plot(list_loss_val, label="loss val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

def plot(x, y_gt, y_pred):
    plt.scatter(x, y_gt, marker ='x', color='blue', s=10, label='y_GT' )
    plt.scatter(x, y_pred, marker ='x', color='green', s=10, label='y_pred' )
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $')
    plt.show()

if __name__ == "__main__":
    main()