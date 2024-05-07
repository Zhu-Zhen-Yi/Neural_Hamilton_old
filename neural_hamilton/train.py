import torch
import torch.nn.functional as F

# def train_epoch(model, optimizer, dataloader, device):
#     model.train()
#     epoch_loss = 0
#     for u, y, Guy in dataloader:  
#         optimizer.zero_grad()
#         pred = model(u.to(device), y.to(device))
#         loss = F.mse_loss(pred, Guy.to(device))
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     epoch_loss /= len(dataloader)
#     return epoch_loss

# def evaluate(model, dataloader, device):
#     model.eval()  
#     eval_loss = 0
#     with torch.no_grad():
#         for u, y, Guy in dataloader:
#             pred = model(u.to(device), y.to(device))
#             loss = F.mse_loss(pred, Guy.to(device)) 
#             eval_loss += loss.item()
#     eval_loss /= len(dataloader)
#     return eval_loss

def train_epoch(model, optimizer, dataloader, device):
    model.train()
    epoch_loss = 0
    epoch_kl_loss = 0
    for u, y, Guy in dataloader:  
        optimizer.zero_grad()
        pred, mu, logvar = model(u.to(device), y.to(device))

        # Flatten
        mu_vec = mu.view((mu.shape[0], -1))             # B, D * L * Z
        logvar_vec = logvar.view((logvar.shape[0], -1)) # B, D * L * Z

        # KL Divergence (mean over latent dimensions)
        kl_loss = -0.5 * torch.mean(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1)
        kl_loss = model.kl_weight * torch.mean(kl_loss)
        loss = F.mse_loss(pred, Guy.to(device)) + kl_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_kl_loss += kl_loss.item()
    epoch_loss /= len(dataloader)
    epoch_kl_loss /= len(dataloader)
    return epoch_loss, epoch_kl_loss

def evaluate(model, dataloader, device):
    model.eval()  
    eval_loss = 0
    eval_kl_loss = 0
    with torch.no_grad():
        for u, y, Guy in dataloader:
            pred, mu, logvar = model(u.to(device), y.to(device))
            
            # Flatten
            mu_vec = mu.view((mu.shape[0], -1))             # B, D * L * Z
            logvar_vec = logvar.view((logvar.shape[0], -1)) # B, D * L * Z

            # KL Divergence (mean over latent dimensions)
            kl_loss = -0.5 * torch.mean(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1)
            kl_loss = model.kl_weight * torch.mean(kl_loss)
            loss = F.mse_loss(pred, Guy.to(device)) + kl_loss
            
            eval_loss += loss.item()
            eval_kl_loss += kl_loss.item()
    eval_loss /= len(dataloader)
    eval_kl_loss /= len(dataloader)
    return eval_loss, eval_kl_loss