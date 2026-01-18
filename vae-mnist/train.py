from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.losses import elbo_loss
from src.model import VAE
from src.utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal VAE MNIST template")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=400)
    p.add_argument("--z-dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outputs", type=str, default="outputs")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def save_reconstructions(model: VAE, batch: torch.Tensor, device: torch.device, out_path: Path, n: int = 16) -> None:
    model.eval()
    x = batch[:n].to(device)
    x_hat_logits, _, _, _ = model(x)
    x_hat = torch.sigmoid(x_hat_logits)
    grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=n)
    save_image(grid, out_path)


@torch.no_grad()
def save_samples(model: VAE, device: torch.device, out_path: Path, n: int = 64) -> None:
    model.eval()
    z = torch.randn(n, model.z_dim, device=device)
    x_hat_logits = model.decode(z)
    x_hat = torch.sigmoid(x_hat_logits)
    grid = make_grid(x_hat.cpu(), nrow=8)
    save_image(grid, out_path)


def run_epoch(model: VAE, loader: DataLoader, optimizer: optim.Optimizer | None, device: torch.device, beta: float) -> Dict[str, float]:
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    out_dir = Path(args.outputs)
    ensure_dir(str(out_dir))

    transform = transforms.ToTensor()
    ds = datasets.MNIST(root=str(out_dir / "data"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(out_dir / "data"), train=False, download=True, transform=transform)

    val_size = 10_000
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VAE(hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fixed_batch, _ = next(iter(val_loader))
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta=args.beta)
        val_metrics = run_epoch(model, val_loader, None, device, beta=args.beta)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} (recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} (recon={val_metrics['recon']:.4f}, kl={val_metrics['kl']:.4f})"
        )

        save_reconstructions(model, fixed_batch, device, out_dir / "recon_grid.png", n=16)
        save_samples(model, device, out_dir / "samples_grid.png", n=64)

    test_metrics = run_epoch(model, test_loader, None, device, beta=args.beta)
    print(f"TEST | loss={test_metrics['loss']:.4f} (recon={test_metrics['recon']:.4f}, kl={test_metrics['kl']:.4f})")

    ckpt_path = out_dir / "vae_mnist.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    fig_path = out_dir / "loss_curve.png"
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("ELBO (per item)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved plot: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
