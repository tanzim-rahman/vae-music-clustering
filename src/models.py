import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=40, latent_dim=16):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Dynamically computes shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 128)
            h = self.encoder(dummy)
            self.flatten_dim = h.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 64, 8, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

class TextEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.fc(z)

class MultiModalVAE(nn.Module):
    def __init__(self, text_dim, latent_dim=32):
        super().__init__()

        # Audio VAE parts
        self.audio_encoder = ConvVAE(latent_dim).encoder
        self.audio_fc_mu = ConvVAE(latent_dim).fc_mu
        self.audio_fc_logvar = ConvVAE(latent_dim).fc_logvar

        self.audio_fc_decode = ConvVAE(latent_dim).fc_decode
        self.audio_decoder = ConvVAE(latent_dim).decoder

        # Text parts
        self.text_encoder = TextEncoder(text_dim, latent_dim)
        self.text_decoder = TextDecoder(latent_dim, text_dim)

    def encode_audio(self, x):
        h = self.audio_encoder(x)
        h = h.view(h.size(0), -1)
        return self.audio_fc_mu(h), self.audio_fc_logvar(h)

    def encode_text(self, x):
        return self.text_encoder(x)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, audio, text):
        mu_a, logvar_a = self.encode_audio(audio)
        mu_t, logvar_t = self.encode_text(text)

        # Fusion (simply takes the mean)
        mu = (mu_a + mu_t) / 2
        logvar = (logvar_a + logvar_t) / 2

        z = self.reparameterise(mu, logvar)

        # Decode
        audio_recon = self.audio_decoder(
            self.audio_fc_decode(z).view(-1, 64, 8, 16)
        )
        text_recon = self.text_decoder(z)

        return audio_recon, text_recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

def multimodal_vae_loss(audio_recon, audio, text_recon, text, mu, logvar):
    recon_audio = F.mse_loss(audio_recon, audio, reduction='mean')
    recon_text = F.mse_loss(text_recon, text, reduction='mean')

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = torch.mean(kl)

    return recon_audio + recon_text + kl
