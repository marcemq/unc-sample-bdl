import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.plot_utils import plot_losses

def train_variational(model, device, train_data, val_data, LEARNING_RATE, EPOCHS, compute_pi=True, mc_its=10):
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
            print(x.shape)
            x = x.to(device).reshape((len(x),1,-1))
            print(x.shape)
            y_gt = y_gt.to(device).reshape((len(y_gt),1,-1))

            # forward pass and losses compute
            pred, nll_loss, kl_loss = model(x, y_gt, mc_its)

            # Compute loss and error
            if compute_pi:
                pi = (2.0**(M-batch_idx))/(2.0**M-1)
            else:
                pi = 1/M
            loss = pi*kl_loss + nll_loss
            error += loss.detach().item()

            # compute gradients and update weights
            loss.backward()
            optimizer.step()
            break
        list_loss_train.append(error/M)

        #validation
        error_val = 0
        M_val = len(val_data)
        model.eval()
        with torch.no_grad():
            for batch_idx, (xval, yval_gt) in enumerate(val_data):
                xval = xval.to(device).reshape((len(xval),1,-1))
                yval_gt = yval_gt.to(device).reshape((len(yval_gt),1,-1))
                pred_val, nll_loss, kl_loss = model(x, y_gt, mc_its)
                if compute_pi:
                    pi = (2.0**(M-batch_idx))/(2.0**M-1)
                else:
                    pi = 1/M
                loss = pi*kl_loss + nll_loss
                error_val += loss.detach().item()
        list_loss_val.append(error/M_val)

    plot_losses(list_loss_train, list_loss_val)

def inference_variational(model, device, test_data, MC_ITS):
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
        # TODO: compute mu and sdt epistemic to be drawn
    return x, y_gt, y_pred

def train_deterministic(model, device, train_data, val_data, LEARNING_RATE, EPOCHS):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999))
    training_losses = []
    validation_losses = []
    # Training
    model.train()
    for epoch in range(EPOCHS):
        batch_train_loss = []
        for x_train, y_train in train_data:
            # Send batch to device
            x_train = x_train.to(device).reshape((len(x_train),-1))
            y_train = y_train.to(device).reshape((len(y_train),-1))
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
            for x_val, y_val in val_data:
                x_val = x_val.to(device).reshape((len(x_val),-1))
                y_val = y_val.to(device).reshape((len(y_val),-1))
                model.eval()
                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                batch_val_loss.append(val_loss.item())
            validation_losses.append(np.mean(batch_val_loss))

        if (epoch+1) % 100 == 0:
            print(f"[{epoch+1}] Training loss: {training_losses[epoch]:.4f}\t Validation loss: {validation_losses[epoch]:.4f}")

def inference_deterministic(model, device, test_data):
    x, y_gt, y_pred = [], [], []
    with torch.no_grad():
        for xi, yi in test_data:
            xi = xi.to(device).reshape((len(xi),-1))
            yi = yi.to(device).reshape((len(yi),-1))
            y_hat = model(xi)
            x.extend(xi.cpu().numpy().tolist())
            y_gt.extend(yi.cpu().numpy().tolist())
            y_pred.extend(y_hat.cpu().numpy().tolist())
    return x, y_gt, y_pred