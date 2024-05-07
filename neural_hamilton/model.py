import torch
from torch import nn

def create_net(sizes):
    net = []
    for i in range(len(sizes)-1):
        net.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            net.append(nn.GELU())
    return nn.Sequential(*net)

class DeepONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.branch_net = create_net([hparams["num_input"]] + [hparams["hidden_size"]]*(hparams["branch_hidden_depth"]-1) + [hparams["num_branch"]]) 
        self.trunk_net = create_net([hparams["dim_output"]] + [hparams["hidden_size"]]*(hparams["trunk_hidden_depth"]-1) + [hparams["num_branch"]])
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, u, y):  
        l = y.shape[1]
        branch_out = self.branch_net(u) 
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1]) for i in range(l)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred
    
class Encoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x):
        """
        - x: (B, W, 1)
        - h_n: (D * L, B, H) (D = 2 for bidirectional)
        - c_n: (D * L, B, H) (D = 2 for bidirectional)
        """
        _, (h_n, c_n) = self.rnn(x)
        return h_n, c_n
    
class Decoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x, h_c):
        """
        - x: (B, W, 1)
        - h_c: (D * L, B, H) (D = 2 for bidirectional)
        - o: (B, W, D * H) (D = 2 for bidirectional)
        - out: (B, W, 1)
        """
        o, _ = self.rnn(x, h_c)
        out = self.fc(o)
        return out

class VAONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hidden_size = hparams["hidden_size"]
        latent_size = hparams["latent_size"]
        num_layers = hparams["num_layers"]
        dropout = hparams["dropout"]
        kl_weight = hparams["kl_weight"]
        
        self.branch_net = Encoder(hidden_size, num_layers, dropout)
        self.trunk_net = Decoder(hidden_size, num_layers, dropout)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        self.fc_z = nn.Linear(latent_size, hidden_size)
        self.kl_weight = kl_weight
        self.reparametrize = True

    def forward(self, u, y):
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        (h0, c0) = self.branch_net(u)

        # Reparameterize (VAE)
        hp = h0.permute(1,0,2).contiguous()     # B, D * L, H
        mu = self.fc_mu(hp)                     # B, D * L, Z
        logvar = self.fc_var(hp)                # B, D * L, Z
        if self.reparametrize:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Decoding
        hz = self.fc_z(z)                       # B, D * L, H
        hzp = hz.permute(1,0,2).contiguous()    # D * L, B, H
        h_c = (hzp, c0)
        o = self.trunk_net(y, h_c)              # B, W2, 1
        return o.squeeze(-1), mu, logvar