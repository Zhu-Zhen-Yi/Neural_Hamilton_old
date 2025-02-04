import torch
from torch.utils.data import DataLoader

# Neural Hamilton modules
from neural_hamilton.model import DeepONet, VAONet, TFONet
from neural_hamilton.data import val_dataset
from neural_hamilton.utils import VAEPredictor, Predictor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import os
import survey
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Choose checkpoint
checkpoints = os.listdir("checkpoints")
chosen = survey.routines.select("Select checkpoint", options=checkpoints)
checkpoint = f"checkpoints/{checkpoints[chosen]}"

# Load the hyperparameters
hparams = json.load(open(f"{checkpoint}/hparams.json", "r"))

# Load the model
## Check if checkpoint contains 'tf' or 'vae'
if "tf" in checkpoint:
    model = TFONet(hparams)
    model.load_state_dict(torch.load(f"{checkpoint}/model.pth"))
    predictor = Predictor(
        model,
        device=device,
        study_name = "DeepONet-Hamilton-Bound",
        run_name = checkpoints[chosen]
    )
elif "vae" in checkpoint:
    model = VAONet(hparams)
    model.load_state_dict(torch.load(f"{checkpoint}/model.pth"))
    predictor = VAEPredictor(
        model,
        device=device,
        study_name = "DeepONet-Hamilton-Bound",
        run_name = checkpoints[chosen]
    )

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
