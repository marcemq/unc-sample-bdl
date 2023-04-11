from matplotlib import pyplot as plt
import numpy as np

def plot_losses(list_loss_train, list_loss_val):
    plt.figure(figsize=(7,7))
    plt.plot(list_loss_train, label="train loss")
    plt.plot(list_loss_val, label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train/Val losses")
    plt.legend()

def plot(x, y_gt, y_pred):
    plt.scatter(x, y_gt, marker ='x', color='blue', s=10, label='y_GT' )
    plt.scatter(x, y_pred, marker ='x', color='green', s=10, label='y_pred' )
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $')

def plotCompleteData(data, dataT, title):
    plt.figure(figsize=(7,7))
    plt.scatter(data[:, 0], data[:, 1], marker ='x', color='blue', s=10)
    plt.scatter(data[:, 0], data[:, 2], marker ='x', color='green', s=10)
    plt.scatter(dataT[:, 0], dataT[:, 1], marker ='x', color='c', s=10, label='unseen data samples')
    plt.scatter(dataT[:, 0], dataT[:, 2], marker ='x', color='green', s=10, label="predicted y")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $', fontsize=12, color='gray')
    plt.suptitle(title, fontsize=16, color='blue')
    plt.legend(loc='best')

def plotVarianceHelper(data, sigma_scale, color='blue', l1="", l2=""):
    sdata = data[np.argsort(data[:, 0])]
    # Ground truth
    plt.scatter(sdata[:, 0], sdata[:, 1], marker ='x', color=color, s=10, label=l1)
    # Prediction
    plt.scatter(sdata[:, 0], sdata[:, 2], marker ='x', color='green', s=10, label=l2)
    # Estimated standard deviation
    y2 = sdata[:, 2] - sigma_scale*sdata[:, 3]
    y3 = sdata[:, 2] + sigma_scale*sdata[:, 3]
    plt.fill_between(sdata[:, 0], y2, y3, color='b', alpha=.1)

def plotWithVariance(data, dataT, sigma_scale, title):
    plt.figure(figsize=(7,7))
    data = data[np.argsort(data[:, 0])]
    idx = np.where(data[:,0] == 1)
    # Predictions corresponding to seen data samples
    dataP1 = data[:idx[0][0]+1,:]
    dataP2 = data[idx[0][0]+1:,:]
    plotVarianceHelper(dataP1, sigma_scale)
    plotVarianceHelper(dataP2, sigma_scale)
    # Predictions corresponding to unseen data samples
    plotVarianceHelper(dataT, sigma_scale, 'c', "Unseen data samples", "Predicted y")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $' + "\n sigma-scale=" + str(sigma_scale), fontsize=12, color='gray')
    plt.suptitle(title, fontsize=16, color='blue')
    plt.legend(loc=2)
