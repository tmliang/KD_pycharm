import numpy as np
import torch
import matplotlib.pyplot as plt

def sigmoid(x, s):
    return 1 / (1 + np.exp(-s*x))

x = np.linspace(-10, 10, 100)
for s in [0.1, 1, 10, 100]:
    y = torch.sigmoid(torch.tensor(s * x)).numpy()
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Sigmoid Function')
    plt.title(f'Sigmoid Function {s}')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
