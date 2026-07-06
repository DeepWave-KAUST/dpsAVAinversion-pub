
import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# -----------------------
# Dataset (NO normalization, dataset should be already normalized)
# -----------------------
class H5MultiChannelDataset(Dataset):
    """
    Expects dataset '<split>' with shape (N, 3, 128, 256), float32.
    Data must already be normalized (e.g. to [-1, 1]).
    """
    def __init__(self, h5_path: str, split: str = "train"):
        self.h5_path = h5_path
        self.split = split
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            if split not in f:
                raise KeyError(f"Split '{split}' not found. Keys: {list(f.keys())}")
            self.length = len(f[split])

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        x = np.asarray(self._h5[self.split][idx], dtype=np.float32)
        assert x.shape == (3, 128, 256), f"Expected (3,128,256), got {x.shape}"
        return torch.from_numpy(x)


# -----------------------
# Model
# -----------------------
def build_model():
    """
    UNet predicts noise ε with the same shape as the input.
    """
    return UNet2DModel(
        sample_size=None,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )




# -----------------------
# Training loop (DDPM)
# -----------------------
def train_loop(
    model,
    noise_scheduler,
    dataloader,
    optimizer,
    device,
    n_epochs=50,
    use_amp=True,
):
    from torch.amp import autocast, GradScaler
    scaler = GradScaler(enabled=use_amp)

    model.train()
    losses = []

    for epoch in range(n_epochs):
        running_loss = 0.0

        for x0 in dataloader:
            # x0: (B,3,128,256)
            x0 = x0.to(device)

            # sample random timestep per sample
            t = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (x0.size(0),),
                device=device,
                dtype=torch.long,
            )

            # creates noise with shape: (B, 3, 128, 256), every channel different noise: Same strategy as training DDPM on RBG images.
            noise = torch.randn_like(x0)
            # add noise
            xt = noise_scheduler.add_noise(x0, noise, t)

            # Resets gradients from the previous iteration
            optimizer.zero_grad(set_to_none=True)

            # Predict the noise and compute the DDPM loss (MSE)
            # 'autocast' enables automatic mixed-precision (AMP):It runs the model mostly in float16 instead of float32 (faster, less memory ecc..)
            with autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                enabled=use_amp,
            ):
                # model’s estimate of the noise ε̂ that was added.
                noise_pred = model(xt, t).sample 
                loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation with gradient scaling: in mixed precision, gradients can underflow (too small).
            # GradScaler scales the loss up before backward, then unscales safely.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - loss: {avg_loss:.6f}")

    return losses


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_seed(123)

    ## Paths:

    h5_path = "/home/brandof/diff/data/avo/elastic_data_upsample_lognorm.h5"  # dataset path
    out_root = Path("/home/brandof/diff/data/avo/checkpoints")             # output path
    run_name = f"3ch-ddpm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"      # output name
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_workers = 4
    n_epochs = 500
    lr = 1e-4

    # Data
    dataset = H5MultiChannelDataset(h5_path, split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        worker_init_fn=_seed_worker,
    )

    # Sanity check
    xb = next(iter(loader))
    print("Batch shape:", xb.shape)
    print("Min / max:", xb.min().item(), xb.max().item())

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # Train
    losses = train_loop(
        model=model,
        noise_scheduler=noise_scheduler,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
        use_amp=True,
    )

    # Save model
    model.save_pretrained(run_dir)
    noise_scheduler.save_pretrained(run_dir / "scheduler")

    # Save metadata
    meta = {
        "h5_path": h5_path,
        "patch_size": [128, 256],
        "channels": ["ch0", "ch1", "ch2"],
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "note": "Simple DDPM training, no padding, no conditioning.",
    }
    with open(run_dir / "training_args.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Training curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("DDPM training loss (3 channels, 128x256)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=150)

    print(f"Training complete. Model saved to:\n{run_dir}")


if __name__ == "__main__":
    main()
