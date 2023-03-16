import torch
import logging
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.mlp_gaussian import MLP_gaussian

# Fixed Hyperparameters
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
EPOCHS = 5

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

def main():

    # set fixed random seed
    torch.manual_seed(42)
    
    # Check current device to work with
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batched_train_data, batched_val_data = getDatasets()

    model = MLP_gaussian(1,10,1).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        for x_batch, y_batch in batched_train_data:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            model.train()
            yhat = model(x_batch)
            loss = loss_fn(yhat, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        # Print loss in the epoch loop only
        print(f"Epoch: {epoch} | Training loss: {np.mean(training_losses):.2f}")


        with torch.no_grad():
            for x_val, y_val in batched_val_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()
                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                validation_losses.append(val_loss.item())

        print(f"[{epoch+1}] Training loss: {np.mean(training_losses):.3f}\t Validation loss: {np.mean(validation_losses):.3f}")

     

if __name__ == "__main__":
    main()