from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from fenn import Fenn
from fenn.utils import set_seed

from modules import VAE, elbo_loss

app = Fenn()


@torch.no_grad()
def save_reconstructions(model, batch, device, out_path, n=16):
    model.eval()
    x = batch[:n].to(device)
    x_hat_logits, _, _, _ = model(x)
    x_hat = torch.sigmoid(x_hat_logits)
    grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=n)
    save_image(grid, out_path)


@torch.no_grad()
def save_samples(model, device, out_path, n=64):
    model.eval()
    z = torch.randn(n, model.z_dim, device=device)
    x_hat_logits = model.decode(z)
    x_hat = torch.sigmoid(x_hat_logits)
    grid = make_grid(x_hat.cpu(), nrow=8)
    save_image(grid, out_path)


def run_epoch(model, loader, optimizer, device, beta):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_items = 0

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    for x, _ in pbar:
        x = x.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        x_hat_logits, mu, logvar, _ = model(x)
        terms = elbo_loss(x_hat_logits, x, mu, logvar, beta=beta)

        if train:
            terms.loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_items += bs
        total_loss += terms.loss.item()
        total_recon += terms.recon.item()
        total_kl += terms.kl.item()

        pbar.set_postfix(
            loss=total_loss / total_items,
            recon=total_recon / total_items,
            kl=total_kl / total_items,
        )

    return {
        "loss": total_loss / total_items,
        "recon": total_recon / total_items,
        "kl": total_kl / total_items,
    }


@app.entrypoint
def main(args):
    # ========================================
    # Setup
    # ========================================
    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    out_dir = Path("outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ========================================
    # Data
    # ========================================
    transform = transforms.ToTensor()

    train_full = datasets.MNIST(
        root=args["dataset"]["dir"], train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=args["dataset"]["dir"], train=False, download=True, transform=transform
    )

    val_size = 10_000
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args["train"]["seed"]),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args["train"]["batch"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args["test"]["batch"], shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args["test"]["batch"], shuffle=False
    )

    # ========================================
    # Model & Optimizer
    # ========================================
    model = VAE(
        hidden_dim=args["model"]["hidden_dim"],
        z_dim=args["model"]["z_dim"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args["train"]["lr"]))

    # ========================================
    # Training Loop
    # ========================================
    fixed_batch, _ = next(iter(val_loader))
    history = {"train_loss": [], "val_loss": []}

    epochs = args["train"]["epochs"]
    beta = float(args["train"]["beta"])

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta=beta)
        val_metrics = run_epoch(model, val_loader, None, device, beta=beta)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"(recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} "
            f"(recon={val_metrics['recon']:.4f}, kl={val_metrics['kl']:.4f})"
        )

        save_reconstructions(model, fixed_batch, device, out_dir / "recon_grid.png")
        save_samples(model, device, out_dir / "samples_grid.png")

    # ========================================
    # Evaluation
    # ========================================
    test_metrics = run_epoch(model, test_loader, None, device, beta=beta)
    print(
        f"TEST | loss={test_metrics['loss']:.4f} "
        f"(recon={test_metrics['recon']:.4f}, kl={test_metrics['kl']:.4f})"
    )

    # ========================================
    # Save Checkpoint & Loss Curve
    # ========================================
    ckpt_path = out_dir / "vae_mnist.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": args}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("ELBO (per item)")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "loss_curve.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    app.run()
