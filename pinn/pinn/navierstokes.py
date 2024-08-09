import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation

nu = 0.01

class NavierStokes():
    def __init__(self, X, Y, t, u, v):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move tensors to the device and set requires_grad for specific variables
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)

        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)
        self.null = torch.zeros((self.x.shape[0], 1), dtype=torch.float32).to(device)

        # Initialize the network
        self.network()

        # Initialize the optimizer
        self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=20000, max_eval=50000,
                                           history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")

        # Define the loss function
        self.loss = nn.MSELoss()

        self.ls = 0
        self.iter = 0

    def network(self):
        # Define the neural network architecture
        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2)
        ).to(self.x.device)  # Move the network to the same device as input tensors

    def function(self, X, Y, t):
        # Predict psi and p using the network
        res = self.net(torch.hstack((X, Y, t)))
        psi, p = res[:, 0:1], res[:, 1:2]

        # Compute the derivatives
        u = torch.autograd.grad(psi, X, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, Y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, Y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, Y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, X, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, Y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, Y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        p_x = torch.autograd.grad(p, X, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, Y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # Compute the residuals of the Navier-Stokes equations
        f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return u, v, p, f, g

    def closure(self):
        self.optimizer.zero_grad()
        u_pred, v_pred, p_pred, f_pred, g_pred = self.function(self.x, self.y, self.t)

        # Compute losses
        u_loss = self.loss(u_pred, self.u)
        v_loss = self.loss(v_pred, self.v)
        f_loss = self.loss(f_pred, self.null)
        g_loss = self.loss(g_pred, self.null)

        # Total loss
        self.ls = u_loss + v_loss + f_loss + g_loss
        self.ls.backward()

        self.iter += 1
        if not self.iter % 1:
            print(f"Iteration {self.iter}, loss: {self.ls.item():.4f}")
        return self.ls

    def train(self):
        self.net.train()
        self.optimizer.step(self.closure)

# Load data
N_train = 5000
data = scipy.io.loadmat("/home/sajid/PycharmProjects/PINN/pinn/cylinder_wake.mat")
U_star = data['U_star']
X_star = data['X_star']
p_star = data['p_star']
t_star = data['t']

# Extract velocity components
u = U_star[:, 0:1, :]
v = U_star[:, 1:2, :]

# Prepare training data
T = t_star.shape[0]
N = X_star.shape[0]
XX = np.tile(X_star[:, 0:1], (1, T))
YY = np.tile(X_star[:, 1:2], (1, T))
TT = np.tile(t_star[:, 0:1], (1, N)).T
x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = u.flatten()[:, None]
v = v.flatten()[:, None]

# Select random training samples
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

# Initialize and train the model
pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
pinn.train()

