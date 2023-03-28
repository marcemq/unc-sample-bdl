import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.mlp_gaussian import MLP_gaussian1, MLP_gaussian2
from matplotlib import pyplot as plt

# Fixed Hyperparameters
BATCH_SIZE = 40
LEARNING_RATE = 1e-3
EPOCHS = 500

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
    # model definition
    model = MLP_gaussian2(1,500,1).to(device)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    training_losses = []
    validation_losses = []
    # Training
    for epoch in range(EPOCHS):
        batch_train_loss = []
        for x_train, y_train in batched_train_data:
            # Send batch to device
            x_train = x_train.to(device).reshape((len(x_train),-1))
            y_train = y_train.to(device).reshape((len(y_train),-1))
            model.train()
            # forward pass
            yhat = model(x_train)
            loss = loss_fn(yhat, y_train)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # save loss per batch
            batch_train_loss.append(loss.item())
        training_losses.append(np.mean(batch_train_loss))

        with torch.no_grad():
            batch_val_loss = []
            for x_val, y_val in batched_val_data:
                x_val = x_val.to(device).reshape((len(x_val),-1))
                y_val = y_val.to(device).reshape((len(y_val),-1))
                model.eval()
                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                batch_val_loss.append(val_loss.item())
            validation_losses.append(np.mean(batch_val_loss))
    
        if (epoch+1) % 10 == 0:
            print(f"[{epoch+1}] Training loss: {training_losses[epoch]:.4f}\t Validation loss: {validation_losses[epoch]:.4f}")

    # inference
    x, y_gt, y_pred = [], [], []
    with torch.no_grad():
        for xi, yi in batched_train_data:
            xi = xi.to(device).reshape((len(xi),-1))
            yi = yi.to(device).reshape((len(yi),-1))
            y_hat = model(xi)
            x.append(xi.numpy())
            y_gt.append(yi.numpy())
            y_pred.append(y_hat.numpy())
    
    plot(x, y_gt, y_pred)

if __name__ == "__main__":
    main()
