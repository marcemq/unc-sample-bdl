from matplotlib import pyplot as plt

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