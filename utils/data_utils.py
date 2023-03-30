import numpy as np
import torch
from os import path
from torch.utils.data import DataLoader, TensorDataset, random_split

# TODO: this function should return train, val and test batchs
# TODO: debug error due to BATCH_SIZE, test with 50 to see the reported error
def getDatasetsTrainVal(BATCH_SIZE, dataFileName="data.csv"):
    data = np.genfromtxt(path.join("data", dataFileName), dtype=float, delimiter=',', names=True) 
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

def getDatasetTestUnseen(BATCH_SIZE, dataFileName="dataUnseen.csv"):
    data = np.genfromtxt(path.join("data", dataFileName), dtype=float, delimiter=',', names=True) 
    x_tensor = torch.from_numpy(data["x"]).float()
    y_tensor = torch.from_numpy(data["y"]).float()

    test_data = TensorDataset(x_tensor, y_tensor)
    batched_test_data = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    return batched_test_data