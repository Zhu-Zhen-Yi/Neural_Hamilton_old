# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RetryFailedTrialCallback
from optuna.visualization import plot_optimization_history, plot_param_importances

# PyTorch for deep learning
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, PolynomialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import lightning as L

# Weights & Biases for experiment tracking
import wandb

# Rich for console output
from rich.console import Console
from rich.progress import Progress

# Neural Hamilton modules
from neural_hamilton.model import DeepONet
from neural_hamilton.data import train_dataset, val_dataset
from neural_hamilton.train import train_epoch, evaluate

import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ds_train = train_dataset()
ds_val = val_dataset()

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepONet Hyperparameter Tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna (default: 100)')
    return parser.parse_args()

# Define the objective function for Optuna
def objective(trial, console, progress, task_id):
    L.seed_everything(42)

    hparams = {
        "num_input": 100,
        "num_branch": trial.suggest_int("num_branch", 1, 9) * 10,
        "num_output": 100,
        "dim_output": 1,
        "hidden_size": 2 ** trial.suggest_int("hidden_size", 5, 9),
        "branch_hidden_depth": trial.suggest_int("branch_hidden_depth", 2, 10),
        "trunk_hidden_depth": trial.suggest_int("trunk_hidden_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": 500,
        "epochs": 200,
    }

    dl_train = DataLoader(ds_train, batch_size=hparams["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=hparams["batch_size"], shuffle=False)

    model = DeepONet(hparams)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = PolynomialLR(optimizer, total_iters=int(hparams["epochs"]), power=2.0)

    checkpoint_dir = trial.study.study_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        run = wandb.init(project="NeuralHamilton-Optuna", config=hparams, reinit=False)
        progress_epoch = progress.add_task("[cyan]Epochs", total=hparams["epochs"])
        for epoch in range(hparams["epochs"]):
            train_loss = train_epoch(model, optimizer, dl_train, device)
            val_loss = evaluate(model, dl_val, device)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1, "lr": scheduler.get_last_lr()[0]})
            scheduler.step()
            progress.update(progress_epoch, advance=1)

            if trial.should_prune():
                raise optuna.TrialPruned()
        progress.update(task_id, advance=1)
    except optuna.TrialPruned:
        run.finish(exit_code=255)
        progress.update(task_id, advance=1)
        raise

    trial_path = os.path.join(checkpoint_dir, f"trial_{trial.number}.pt")
    torch.save(model.state_dict(), trial_path)
    trial.set_user_attr("checkpoint", trial_path)

    run.finish()

    return val_loss

def main():
    args = parse_arguments()

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=5)

    console = Console()
    progress = Progress()

    with progress:
        task_id = progress.add_task("[green]Optuna trials", total=args.n_trials)

        study = optuna.create_study(
            study_name="NeuralHamilton",
            storage="sqlite:///NeuralHamilton.db",
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=True
            )
        study.optimize(lambda trial: objective(trial, console, progress, task_id), n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()