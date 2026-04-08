import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(BetaVAE, self).__init__()

        # encoder: input -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 128(encoder) -> mu, logvar -> 128(decoder)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        # decoder: 128 -> 256 -> output(=input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    # encoder returns mu and logvar
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    # reparameterization trick (z = mu + eps*std)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # decoder takes in z and outputs reconstruction
    def decode(self, z):
        return self.decoder(z)

    # forward pass of full model
    # returns reconstruction, mu and logvar
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# calculating two losses: reconstruction loss and kl divergence
# reconstruction loss: mse between input and reconstruction
# kl divergence: uses mu and logvar
# final loss: weighted sum of the two, where beta=1 for VAE
# for beta vae, beta > 1
def beta_vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = F.mse_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
