from matplotlib import pyplot as plt

def plot_losses(list_loss_train, list_loss_val):
    plt.figure(figsize=(8,8))
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

def plotComplete(x, y_gt, y_pred, xU, yU_gt, yU_pred):
    plt.scatter(x, y_gt, marker ='x', color='blue', s=10)
    plt.scatter(x, y_pred, marker ='x', color='green', s=10)
    plt.scatter(xU, yU_gt, marker ='x', color='c', s=10, label='unseen data samples')
    plt.scatter(xU, yU_pred, marker ='x', color='green', s=10, label="predicted y")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $')
    plt.legend(loc='best')