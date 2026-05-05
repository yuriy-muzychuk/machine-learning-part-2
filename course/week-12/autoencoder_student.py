"""
autoencoder_student.py — Autoencoder (Student Version)
=======================================================
Week 12 Lab: Autoencoder implementation and latent space exploration.

Classes / functions defined here
---------------------------------
Encoder                — maps input x to latent code z
Decoder                — maps latent code z back to reconstruction x_hat
Autoencoder            — combines Encoder + Decoder (same interface as lecture week 11)
DenoisingAutoencoder   — Autoencoder that corrupts input during training
train_autoencoder      — training loop with MSE loss and Adam optimiser
reconstruction_error   — per-sample MSE between input and reconstruction (given)

Methods YOU implement  (6 TODOs)
---------------------------------
Encoder.forward
Decoder.forward
Autoencoder.encode
Autoencoder.forward
DenoisingAutoencoder.forward
train_autoencoder

Instructions
------------
Implement every method marked with
    raise NotImplementedError("TODO: ...")

Step-by-step comments inside each method guide you.
Do NOT import anything beyond what is already imported.
Do NOT change method signatures or __init__ constructors.

Tip: the lecture (week 11) showed an identical Autoencoder on digits.
     Use the same structural logic; only the forward methods need filling in.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Section 1 — Encoder
# =============================================================================

class Encoder(nn.Module):
    """Maps input x to latent representation z.

    Architecture
    ------------
    x -> Linear(input_dim, hidden_dim) -> ReLU
      -> Linear(hidden_dim, latent_dim) -> ReLU
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to latent code z.

        Parameters
        ----------
        x : Tensor (batch, input_dim)

        Returns
        -------
        z : Tensor (batch, latent_dim)

        Steps
        -----
        1. First layer + activation:
               h = self.act(self.fc1(x))
        2. Second layer + activation:
               z = self.act(self.fc2(h))
        3. Return z.
        """
        raise NotImplementedError("TODO 1: implement Encoder.forward")


# =============================================================================
# Section 2 — Decoder
# =============================================================================

class Decoder(nn.Module):
    """Maps latent code z back to reconstruction x_hat.

    Architecture
    ------------
    z -> Linear(latent_dim, hidden_dim) -> ReLU
      -> Linear(hidden_dim, output_dim) -> Sigmoid
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1     = nn.Linear(latent_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.act     = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code z to reconstruction x_hat.

        Parameters
        ----------
        z : Tensor (batch, latent_dim)

        Returns
        -------
        x_hat : Tensor (batch, output_dim) — values in [0, 1] due to Sigmoid

        Steps
        -----
        1. First layer + ReLU:
               h = self.act(self.fc1(z))
        2. Second layer + Sigmoid (output values in [0, 1]):
               x_hat = self.out_act(self.fc2(h))
        3. Return x_hat.
        """
        raise NotImplementedError("TODO 2: implement Decoder.forward")


# =============================================================================
# Section 3 — Autoencoder
# =============================================================================

class Autoencoder(nn.Module):
    """Symmetric fully-connected autoencoder: Encoder + Decoder.

    Interface identical to the lecture (week 11) demo; generalised to
    arbitrary input_dim so it works for Fashion-MNIST (784-D) as well
    as the lecture's digits (64-D).

    Parameters
    ----------
    input_dim  : int — dimensionality of raw input
    hidden_dim : int — width of hidden layers in encoder and decoder
    latent_dim : int — bottleneck size
    """

    def __init__(
        self,
        input_dim:  int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation z for input x.

        Steps
        -----
        1. Return self.encoder(x).
        """
        raise NotImplementedError("TODO 3: implement Autoencoder.encode")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent code z. Already implemented — do not change."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode. Returns x_hat.

        Steps
        -----
        1. z     = self.encode(x)
        2. x_hat = self.decode(z)
        3. Return x_hat.
        """
        raise NotImplementedError("TODO 4: implement Autoencoder.forward")


# =============================================================================
# Section 4 — Denoising Autoencoder
# =============================================================================

class DenoisingAutoencoder(Autoencoder):
    """Autoencoder trained to reconstruct clean input from a corrupted version.

    During training, forward() adds Gaussian noise to x before encoding.
    The MSE loss is computed against the original clean x by the caller,
    so the model learns a denoising mapping as a side-effect of reconstruction.
    During evaluation (.eval() mode) no noise is added.

    This is the same idea as the Denoising AE from the lecture (week 11, section 6),
    applied here to Fashion-MNIST with a larger architecture.

    Parameters
    ----------
    input_dim, hidden_dim, latent_dim : see Autoencoder
    noise_std : float — standard deviation of Gaussian noise (training only)
    """

    def __init__(
        self,
        input_dim:  int   = 784,
        hidden_dim: int   = 256,
        latent_dim: int   = 16,
        noise_std:  float = 0.2,
    ):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupt x during training, then encode and decode.

        Parameters
        ----------
        x : Tensor (batch, input_dim) — clean input

        Returns
        -------
        x_hat : Tensor (batch, input_dim) — reconstruction (of clean x)

        Steps
        -----
        1. If self.training is True, add Gaussian noise and clamp to [0, 1]:
               noise = torch.randn_like(x) * self.noise_std
               x     = torch.clamp(x + noise, 0.0, 1.0)
           (If self.training is False, leave x unchanged — no noise at eval time.)

        2. Encode the (possibly noisy) x:
               z = self.encode(x)

        3. Decode to get reconstruction:
               x_hat = self.decode(z)

        4. Return x_hat.
           Note: the caller computes MSE(x_hat, original_clean_x).
        """
        raise NotImplementedError("TODO 5: implement DenoisingAutoencoder.forward")


# =============================================================================
# Section 5 — Training
# =============================================================================

def train_autoencoder(
    model:      Autoencoder,
    X_train:    np.ndarray,
    n_epochs:   int   = 100,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    seed:       int   = 42,
) -> list:
    """Train the autoencoder with MSE reconstruction loss (Adam optimiser).

    Works for both Autoencoder and DenoisingAutoencoder: the DAE adds noise
    internally during forward; the MSE is always computed against the clean batch.

    Parameters
    ----------
    model      : Autoencoder or DenoisingAutoencoder
    X_train    : ndarray (n, input_dim) — values in [0, 1]
    n_epochs   : int
    lr         : float — Adam learning rate
    batch_size : int
    seed       : int

    Returns
    -------
    losses : list[float] — mean MSE per epoch

    Steps
    -----
    1. Set random seed:
           torch.manual_seed(seed)

    2. Build DataLoader:
           X_t     = torch.tensor(X_train, dtype=torch.float32)
           dataset = TensorDataset(X_t)
           loader  = DataLoader(
               dataset, batch_size=batch_size, shuffle=True,
               generator=torch.Generator().manual_seed(seed),
           )

    3. Define loss and optimiser:
           criterion = nn.MSELoss()
           optimizer = optim.Adam(model.parameters(), lr=lr)

    4. Set model to training mode:
           model.train()

    5. Initialise losses = [].

    6. Loop for n_epochs; for each epoch:
         a. epoch_loss = 0.0
         b. For each (batch,) in loader:
                optimizer.zero_grad()
                x_hat = model(batch)           # DAE corrupts batch internally
                loss  = criterion(x_hat, batch)  # compare to clean batch
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
         c. losses.append(epoch_loss / len(X_train))

    7. Return losses.
    """
    raise NotImplementedError("TODO 6: implement train_autoencoder")


# =============================================================================
# Section 6 — Evaluation helper  (already implemented)
# =============================================================================

def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error.

    Parameters
    ----------
    model : Autoencoder (already trained)
    X     : ndarray (n, input_dim)

    Returns
    -------
    errors : ndarray (n,) — MSE per sample
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32)
        x_hat = model(X_t).numpy()
    return np.mean((X - x_hat) ** 2, axis=1)
