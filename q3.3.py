import torch
import matplotlib.pyplot as plt
import numpy as np

# Define 2D function: f(x, y) = x^2 + y^2 (minimization) or -x^2 - y^2 (maximization)
def f(xy, maximize=False):
    x, y = xy[0], xy[1]
    val = x**2 + y**2
    return -val if maximize else val

# For plotting contour
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z_min = X**2 + Y**2
Z_max = -X**2 - Y**2

# Run SGD and record path
def run_sgd(momentum=0.0, weight_decay=0.0, maximize=False, steps=50):
    xy = torch.tensor([2.0, 2.0], requires_grad=True)
    opt = torch.optim.SGD([xy], lr=0.1, momentum=momentum, weight_decay=weight_decay, maximize=maximize)
    path = []
    for _ in range(steps):
        opt.zero_grad()
        loss = f(xy, maximize)
        loss.backward()
        opt.step()
        path.append(xy.detach().numpy().copy())
    return np.array(path)

# Plot helper
def plot_path(path, title, maximize=False):
    plt.figure(figsize=(5, 4))
    Z = Z_max if maximize else Z_min
    cp = plt.contour(X, Y, Z, levels=20)
    plt.plot(path[:, 0], path[:, 1], marker='o', color='red')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# (a) Momentum effect
path_m0 = run_sgd(momentum=0.0)
path_m9 = run_sgd(momentum=0.9)
plot_path(path_m0, "SGD: momentum = 0")
plot_path(path_m9, "SGD: momentum = 0.9")

# (b) Add weight decay
path_wd = run_sgd(momentum=0.9, weight_decay=0.1)
plot_path(path_wd, "SGD: momentum = 0.9, weight_decay = 0.1")

# (c) Maximization
path_max = run_sgd(momentum=0.0, weight_decay=0.0, maximize=True)
plot_path(path_max, "SGD: maximize = True, momentum = 0")
