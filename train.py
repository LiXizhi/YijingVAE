"""Training framework for the 6-latent Beta-VAE.

Can be invoked from the CLI:
    python train.py --epochs 20 --beta 4.0 --dataset mnist

Or driven from the web UI via TrainingManager (see server.py).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.optim as optim

from model import BetaVAE, beta_vae_loss, LATENT_DIM
from datasets import get_dataloaders


ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TrainConfig:
    dataset: str = "stl10"        # 64x64 RGB real-world photos
    data_root: str = os.path.join(ROOT, "data")
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    beta: float = 4.0
    hidden: int = 64
    image_size: int = 64
    num_workers: int = 0
    out_dir: str = os.path.join(ROOT, "checkpoints")
    device: str = "auto"          # use GPU when available; otherwise CPU


@dataclass
class TrainState:
    running: bool = False
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    steps_per_epoch: int = 0
    loss: float = 0.0
    recon: float = 0.0
    kld: float = 0.0
    history: list = field(default_factory=list)  # list of {epoch, loss, recon, kld}
    message: str = "idle"
    config: dict = field(default_factory=dict)
    # Live latent stats from the most recent training batch (per-dim mean of mu).
    latent_mu: list = field(default_factory=list)
    latent_logvar: list = field(default_factory=list)


def _resolve_device(name: str) -> torch.device:
    name = (name or "cuda").lower()
    if name in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


class TrainingManager:
    """Runs training in a background thread; exposes live state for the web UI."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state = TrainState()
        self.model: Optional[BetaVAE] = None
        self.device: Optional[torch.device] = None
        self.ckpt_path: Optional[str] = None

    # ---------- public API ----------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: TrainConfig) -> bool:
        if self.is_running():
            return False
        self._stop.clear()
        self.state = TrainState(running=True, total_epochs=cfg.epochs,
                                message="starting", config=asdict(cfg))
        self._thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running():
            return False
        self._stop.set()
        with self._lock:
            self.state.message = "stopping"
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self.state)

    def load_latest(self, out_dir: str = "./checkpoints",
                    in_channels: int = 3, image_size: int = 64,
                    hidden: int = 64) -> bool:
        path = os.path.join(out_dir, "latest.pt")
        if not os.path.exists(path):
            return False
        device = _resolve_device("cuda")
        ckpt = torch.load(path, map_location=device)
        meta = ckpt.get("meta", {})
        model = BetaVAE(
            in_channels=meta.get("in_channels", in_channels),
            image_size=meta.get("image_size", image_size),
            latent_dim=LATENT_DIM,
            hidden=meta.get("hidden", hidden),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.device = device
        self.ckpt_path = path
        return True

    @torch.no_grad()
    def decode(self, z_list):
        if self.model is None:
            raise RuntimeError("No model loaded.")
        z = torch.tensor([z_list], dtype=torch.float32, device=self.device)
        x = self.model.decode(z)[0].clamp(0, 1).cpu().numpy()
        return x  # (C,H,W) in [0,1]

    # ---------- worker ----------
    def _run(self, cfg: TrainConfig):
        try:
            os.makedirs(cfg.out_dir, exist_ok=True)
            device = _resolve_device(cfg.device)
            with self._lock:
                self.state.message = f"preparing {cfg.dataset} from {cfg.data_root}"
            train_loader, _, in_channels, image_size = get_dataloaders(
                cfg.dataset, batch_size=cfg.batch_size,
                num_workers=cfg.num_workers, image_size=cfg.image_size,
                data_root=cfg.data_root)

            with self._lock:
                self.state.message = "building model"
            model = BetaVAE(in_channels=in_channels, image_size=image_size,
                            latent_dim=LATENT_DIM, hidden=cfg.hidden).to(device)
            opt = optim.Adam(model.parameters(), lr=cfg.lr)
            # Expose the live model so the web UI can query its tensor shapes
            # while training is in progress.
            self.model = model
            self.device = device

            steps_per_epoch = len(train_loader)
            with self._lock:
                self.state.steps_per_epoch = steps_per_epoch
                self.state.message = f"training on {device}"

            for epoch in range(1, cfg.epochs + 1):
                if self._stop.is_set():
                    break
                model.train()
                ep_loss = ep_recon = ep_kld = 0.0
                for step, (x, _) in enumerate(train_loader, start=1):
                    if self._stop.is_set():
                        break
                    x = x.to(device)
                    x_hat, mu, logvar, _ = model(x)
                    # reshape target to match x_hat (model outputs original size)
                    loss, recon, kld = beta_vae_loss(x_hat, x, mu, logvar, beta=cfg.beta)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    ep_loss  += loss.item()
                    ep_recon += recon.item()
                    ep_kld   += kld.item()

                    with self._lock:
                        self.state.epoch = epoch
                        self.state.step = step
                        self.state.loss = loss.item()
                        self.state.recon = recon.item()
                        self.state.kld = kld.item()
                        # Update latent stats every few steps to keep cost low.
                        if step % 5 == 0 or step == 1:
                            self.state.latent_mu = mu.detach().mean(0).cpu().tolist()
                            self.state.latent_logvar = logvar.detach().mean(0).cpu().tolist()

                n = max(1, step)
                entry = {"epoch": epoch,
                         "loss":  ep_loss  / n,
                         "recon": ep_recon / n,
                         "kld":   ep_kld   / n}
                with self._lock:
                    self.state.history.append(entry)
                    self.state.message = f"epoch {epoch} done"

                # save checkpoint each epoch
                ckpt = {
                    "model": model.state_dict(),
                    "meta": {
                        "in_channels": in_channels,
                        "image_size":  image_size,
                        "hidden":      cfg.hidden,
                        "latent_dim":  LATENT_DIM,
                        "dataset":     cfg.dataset,
                        "beta":        cfg.beta,
                        "epoch":       epoch,
                    },
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "latest.pt"))
                with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
                    json.dump(self.state.history, f, indent=2)

            # auto-load the trained model for inference
            self.load_latest(cfg.out_dir, in_channels, image_size, cfg.hidden)
            with self._lock:
                self.state.message = "finished" if not self._stop.is_set() else "stopped"
        except Exception as e:
            with self._lock:
                self.state.message = f"error: {e}"
        finally:
            with self._lock:
                self.state.running = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stl10",
                   choices=["stl10", "cifar10", "mnist", "fashion"])
    p.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--beta",    type=float, default=4.0)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--device",  default="auto")
    args = p.parse_args()

    cfg = TrainConfig(dataset=args.dataset, data_root=args.data_root, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, beta=args.beta,
                      hidden=args.hidden, image_size=args.image_size,
                      num_workers=args.num_workers,
                      out_dir=args.out_dir, device=args.device)

    mgr = TrainingManager()
    mgr.start(cfg)
    while mgr.is_running():
        s = mgr.snapshot()
        print(f"[{s['message']}] epoch {s['epoch']}/{s['total_epochs']} "
              f"step {s['step']}/{s['steps_per_epoch']} "
              f"loss={s['loss']:.3f} recon={s['recon']:.3f} kld={s['kld']:.3f}",
              end="\r", flush=True)
        time.sleep(1.0)
    print("\n", mgr.snapshot()["message"])


if __name__ == "__main__":
    main()
