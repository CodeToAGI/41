
"""
EP41 Challenge — Train a VAE on MNIST + Latent Interpolation
pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ── Hyperparameters ──────────────────────────────────────────────────────────
LATENT_DIM = 20
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),          # [0,1]
])
train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1      = nn.Linear(784, 400)
        self.fc_mu    = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# ── Train ────────────────────────────────────────────────────────────────────
model = VAE(LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Training on {DEVICE} …")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.view(-1, 784).to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss = {avg:.1f}")

torch.save(model.state_dict(), "vae_mnist.pt")
print("Model saved → vae_mnist.pt")

# ── 1. Generate new digits from pure noise ───────────────────────────────────
model.eval()
with torch.no_grad():
    z = torch.randn(64, LATENT_DIM).to(DEVICE)
    samples = model.decoder(z).cpu().view(-1, 28, 28)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Generated digits (z ~ N(0,1))", fontsize=14)
plt.tight_layout()
plt.savefig("generated_digits.png", dpi=150)
print("Saved → generated_digits.png")

# ── 2. Latent interpolation between two real digits ──────────────────────────
# Pick two digits from the test set (e.g. a 3 and an 8)
with torch.no_grad():
    # find one example of digit 3 and one of digit 8
    x3, x8 = None, None
    for x, y in test_loader:
        for img, label in zip(x, y):
            if label == 3 and x3 is None:
                x3 = img.view(1, 784).to(DEVICE)
            if label == 8 and x8 is None:
                x8 = img.view(1, 784).to(DEVICE)
            if x3 is not None and x8 is not None:
                break
        if x3 is not None and x8 is not None:
            break

    mu3, _ = model.encoder(x3)
    mu8, _ = model.encoder(x8)

    steps = 10
    alphas = torch.linspace(0, 1, steps).to(DEVICE)
    interps = []
    for a in alphas:
        z = (1 - a) * mu3 + a * mu8          # linear interpolation
        recon = model.decoder(z).cpu().view(28, 28)
        interps.append(recon)

fig, axes = plt.subplots(1, steps, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(interps[i], cmap="gray")
    ax.axis("off")
    ax.set_title(f"{i/(steps-1):.1f}", fontsize=8)
plt.suptitle("Latent interpolation: digit 3 → digit 8", fontsize=12)
plt.tight_layout()
plt.savefig("interpolation.png", dpi=150)
print("Saved → interpolation.png")

print("\n✅ Challenge complete! Post generated_digits.png + interpolation.png in the comments.")
