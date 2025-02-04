# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RetryFailedTrialCallback

# PyTorch for deep learning
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
import lightning as L

# Weights & Biases for experiment tracking
import wandb

# Rich for console output
from rich.console import Console
from rich.progress import Progress

# Neural Hamilton modules
from neural_hamilton.model import DeepONet, VAONet
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

    # hparams = {
    #     "num_input": 100,
    #     "num_branch": trial.suggest_int("num_branch", 1, 4) * 10,
    #     "num_output": 100,
    #     "dim_output": 1,
    #     "hidden_size": 2 ** trial.suggest_int("hidden_size", 5, 7),
    #     "branch_hidden_depth": trial.suggest_int("branch_hidden_depth", 2, 10),
    #     "trunk_hidden_depth": trial.suggest_int("trunk_hidden_depth", 2, 10),
    #     "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
    #     "batch_size": 5000,
    #     "epochs": 200,
    # }
    hparams = {
        "hidden_size": 86,
        "num_layers": 4,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "batch_size": 1000,
        "epochs": 500,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "latent_size": 14,
        "kl_weight": trial.suggest_float("kl_weight", 1e-3, 1e-2, log=True),
    }

    dl_train = DataLoader(ds_train, batch_size=hparams["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=hparams["batch_size"], shuffle=False)

    # model = DeepONet(hparams)
    model = VAONet(hparams)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = PolynomialLR(optimizer, total_iters=int(hparams["epochs"]), power=2.0)

    checkpoint_dir = trial.study.study_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        run = wandb.init(project="NeuralHamilton-VAE(500)", config=hparams, reinit=False)
        progress_epoch = progress.add_task("[cyan]Epochs", total=hparams["epochs"])
        for epoch in range(hparams["epochs"]):
            train_loss, train_kl_loss = train_epoch(model, optimizer, dl_train, device)
            val_loss, val_kl_loss = evaluate(model, dl_val, device)
            scheduler.step()
            progress.update(progress_epoch, advance=1)

            trial.report(val_loss, epoch)

            wandb.log({
                "train_loss": train_loss,
                "train_kl_loss": train_kl_loss,
                "val_loss": val_loss,
                "val_kl_loss": val_kl_loss,
                "epoch": epoch+1, 
                "lr": scheduler.get_last_lr()[0]
            })

            if trial.should_prune():
                raise optuna.TrialPruned()
            
        progress.remove_task(progress_epoch)
        progress.update(task_id, advance=1)
    except optuna.TrialPruned:
        run.finish(exit_code=255)
        progress.remove_task(progress_epoch)
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
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    console = Console()
    progress = Progress()

    with progress:
        task_id = progress.add_task("[green]Optuna trials", total=args.n_trials)

        study = optuna.create_study(
            study_name="NeuralHamilton-VAE(500)",
            storage="sqlite:///NeuralHamilton2.db",
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
