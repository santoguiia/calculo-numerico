import numpy as np
import matplotlib.pyplot as plt

def sinc(x):
    return np.sin(x) / x

def plot_sinc(x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='sinc(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title("Função sinc(x)")
    plt.xlabel("x")
    plt.ylabel("sinc(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

# plot
x = np.linspace(-100, 100, 500)
y = sinc(x)
plot_sinc(x, y)