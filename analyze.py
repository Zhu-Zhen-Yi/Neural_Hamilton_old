# Optuna for hyperparameter tuning
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import torch
from torch.utils.data import DataLoader

# Neural Hamilton modules
from neural_hamilton.model import DeepONet, VAONet
from neural_hamilton.data import train_dataset, val_dataset
from neural_hamilton.train import train_epoch, evaluate
from neural_hamilton.utils import VAEPredictor

# Plotly for visualization
from plotly.offline import plot
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the best study
study = optuna.load_study(study_name="NeuralHamilton-VAE", storage="sqlite:///NeuralHamilton.db")
best_trial = study.best_trial
checkpoint = best_trial.user_attrs["checkpoint"]
print(f"Best trial: {best_trial.number}")
print(f"Best value: {best_trial.value}")
print(f"Best parameters: {best_trial.params}")
print(f"Checkpoint: {checkpoint}")

# Load the best hyperparameters
hparams = best_trial.params
hparams["dropout"] = 0.1

# Load the model
model = VAONet(hparams)
model.load_state_dict(torch.load(checkpoint))

# Create the predictor
predictor = VAEPredictor(model, device=device, study_name = "NeuralHamilton-VAE", run_name = f"{best_trial.number}")

# ==============================================================================
# Validation dataset
# ==============================================================================
# Load the data
ds_val = val_dataset()
u, y, Guy = ds_val[0]
x = np.linspace(0, 1, len(u))

# Plot the potential
predictor.potential_plot(x, u, name="potential_val")

# Plot the prediction
predictor.predict_plot(u, y, Guy, name="prediction_val")

# ==============================================================================
# Simple Harmonic Oscillator
# ==============================================================================
# Create a simple harmonic oscillator dataset
x = torch.linspace(0, 1, 100)
u = 8 * (x - 0.5) ** 2
y = torch.linspace(0, 2, 100)
Guy = 0.5 - 0.5 * torch.cos(4 * y)

# Plot the potential
predictor.potential_plot(x, u, name="potential_sho")

# Plot the prediction
predictor.predict_plot(u, y, Guy, name="prediction_sho")

# ==============================================================================
# Quartic potential
# ==============================================================================
# ODE solve
from scipy.integrate import odeint

def V(x):
    return 625/8 * (x - 1/5)**2 * (x - 4/5)**2

x = np.linspace(0, 1, 100)
def dydt(y, t):
    x, p = y
    dxdt = p
    dpdt = -625/2 * x**3 + 1875/4 * x**2 - 825/4 * x + 25
    return [dxdt, dpdt]

y0 = [0, 0]  # 초기 조건: x(0) = 0, x'(0) = 0
t = np.linspace(0, 2, 100)  # 시간 범위 설정
sol = odeint(dydt, y0, t)

x_true = sol[:,0]

# Create a quartic potential dataset
x = torch.linspace(0, 1, 100)
u = V(x)
y = torch.linspace(0, 2, 100)
Guy = torch.tensor(x_true)

# Plot the potential
predictor.potential_plot(x, u, name="potential_quartic")

# Plot the prediction
predictor.predict_plot(u, y, Guy, name="prediction_quartic")