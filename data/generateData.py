import numpy as np
from matplotlib import pyplot as plt

def f(samples=100, sigma=0.002):
    mu = 0
    e = np.random.normal(mu, sigma, samples*2).reshape(samples*2, 1)
    x1 = np.linspace(0, 1, samples).reshape(samples,1)
    x2 = np.linspace(1.5, 2.5, samples).reshape(samples,1)
    x = np.concatenate([x1,x2])
    y = x + 0.3*np.sin(2*np.pi*(x+e)) + 0.3*np.sin(4*np.pi*(x+e)) + e
    return x, y

def fNew(samples=100, sigma=0.002):
    mu = 0
    e = np.random.normal(mu, sigma, samples).reshape(samples, 1)
    x = np.linspace(1, 1.5, samples).reshape(samples,1)
    y = x + 0.3*np.sin(2*np.pi*(x+e)) + 0.3*np.sin(4*np.pi*(x+e)) + e
    return x, y

def saveData(x, y, fileName="data.csv"):
    data = np.concatenate([x,y], axis=1)
    np.savetxt(fileName, data, delimiter=',', header="x,y")

if __name__ == "__main__":
    x, y = f(samples=200, sigma=0.03)
    xUnseen, yUnseen = fNew(sigma=0.03)
    plt.scatter(x, y, marker ='x', color='b', s=10)
    plt.scatter(xUnseen, yUnseen, marker ='x', color='c', s=10)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(r'$f(x) = x + 0.3\sin(2\pi(x+e)) + 0.3\sin(4\pi(x+e)) + e $')
    plt.show()

    saveData(x,y, "dataS03.csv")
    saveData(xUnseen,yUnseen, "dataUnseenS03.csv")