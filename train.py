"""Training framework for the 6-latent Beta-VAE.

Can be invoked from the CLI:
    python train.py --epochs 20 --beta 4.0 --dataset mnist

Or driven from the web UI via TrainingManager (see server.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import torch
import torch.optim as optim
from PIL import Image
import numpy as np

from model import BetaVAE, beta_vae_loss, LATENT_DIM
from datasets import get_dataloaders


ROOT = os.path.dirname(os.path.abspath(__file__))


def _encode_sample_png(t: torch.Tensor) -> str:
    """Encode a single (C,H,W) tensor in [0,1] as a base64 PNG data URL."""
    arr = t.detach().clamp(0, 1).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode="L")
    else:
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class TrainConfig:
    dataset: str = "stl10"        # 64x64 RGB real-world photos
    data_root: str = os.path.join(ROOT, "data")
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    beta: float = 4.0
    # 128 channels (~5.9 M params at 64×64×3) is the smallest size that
    # reliably learns STL-10 / CIFAR-10. Smaller datasets override this
    # via the per-dataset defaults in the web UI.
    hidden: int = 128
    image_size: int = 64
    num_workers: int = 0
    out_dir: str = os.path.join(ROOT, "checkpoints")
    device: str = "auto"          # use GPU when available; otherwise CPU
    resume: bool = False          # continue from <out_dir>/<dataset>/latest.pt


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
    # Most recent reconstruction sample (data: URL of a PNG) for the UI viz.
    sample_image: str = ""
    # Most recent input image (the x that produced sample_image), so the
    # 3D viz can paint both ends of the encoder–decoder pipeline.
    input_image: str = ""
    # UI-tunable sample-image emit rate, in Hz (1–30).
    preview_fps: float = 2.0
    # Where the active checkpoint lives.
    checkpoint_path: str = ""
    checkpoint_dataset: str = ""


def _resolve_device(name: str) -> torch.device:
    name = (name or "cuda").lower()
    if name in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


# ---------- per-dataset checkpoint layout ----------
def dataset_dir(out_dir: str, dataset: str) -> str:
    return os.path.join(out_dir, dataset)


def checkpoint_path(out_dir: str, dataset: str) -> str:
    return os.path.join(dataset_dir(out_dir, dataset), "latest.pt")


def info_path(out_dir: str, dataset: str) -> str:
    return os.path.join(dataset_dir(out_dir, dataset), "info.json")


def list_checkpoints(out_dir: str) -> list:
    """Return a list of {dataset, info} for every saved checkpoint."""
    items = []
    if not os.path.isdir(out_dir):
        return items
    for name in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, name)
        if not os.path.isdir(sub):
            continue
        ckpt = os.path.join(sub, "latest.pt")
        if not os.path.exists(ckpt):
            continue
        info = {}
        info_file = os.path.join(sub, "info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
        info.setdefault("dataset", name)
        info["checkpoint"] = ckpt
        items.append({"dataset": name, "info": info})
    return items


def _build_info(cfg: "TrainConfig", in_channels: int, image_size: int,
                history: list, current_epoch: int,
                last_loss: float, last_recon: float, last_kld: float,
                param_counts: Optional[dict] = None) -> dict:
    return {
        "dataset": cfg.dataset,
        "config": asdict(cfg),
        "in_channels": in_channels,
        "image_size": image_size,
        "hidden": cfg.hidden,
        "latent_dim": LATENT_DIM,
        "beta": cfg.beta,
        "epochs_completed": current_epoch,
        "epochs_planned": cfg.epochs,
        "history": history,
        "last_loss": float(last_loss),
        "last_recon": float(last_recon),
        "last_kld": float(last_kld),
        "param_counts": param_counts or {},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


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
        self.ckpt_dataset: Optional[str] = None
        self.ckpt_info: dict = {}
        # Sample-image throttle (Hz). UI-tunable in [1, 30].
        self._preview_fps: float = 2.0

    def set_preview_fps(self, fps: float) -> float:
        fps = max(1.0, min(30.0, float(fps)))
        self._preview_fps = fps
        with self._lock:
            self.state.preview_fps = fps
        return fps

    # ---------- public API ----------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: TrainConfig) -> bool:
        if self.is_running():
            return False
        self._stop.clear()
        self.state = TrainState(running=True, total_epochs=cfg.epochs,
                                message="starting", config=asdict(cfg),
                                preview_fps=self._preview_fps)
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
            snap = asdict(self.state)
        snap["checkpoint_info"] = dict(self.ckpt_info)
        return snap

    def load_checkpoint(self, path: str, dataset: Optional[str] = None) -> bool:
        """Load a specific checkpoint .pt file plus its sidecar info.json."""
        if not path or not os.path.exists(path):
            return False
        device = _resolve_device("cuda")
        ckpt = torch.load(path, map_location=device)
        meta = ckpt.get("meta", {})
        model = BetaVAE(
            in_channels=meta.get("in_channels", 3),
            image_size=meta.get("image_size", 64),
            latent_dim=LATENT_DIM,
            hidden=meta.get("hidden", 64),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.device = device
        self.ckpt_path = path
        self.ckpt_dataset = dataset or meta.get("dataset")
        info_file = os.path.join(os.path.dirname(path), "info.json")
        info: dict = {}
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
        info.setdefault("dataset", self.ckpt_dataset or "(unknown)")
        info.setdefault("checkpoint", path)
        info.setdefault("epochs_completed", meta.get("epoch"))
        info.setdefault("hidden", meta.get("hidden"))
        info.setdefault("image_size", meta.get("image_size"))
        info.setdefault("in_channels", meta.get("in_channels"))
        info.setdefault("beta", meta.get("beta"))
        info["loaded_at"] = datetime.now().isoformat(timespec="seconds")
        self.ckpt_info = info
        with self._lock:
            self.state.checkpoint_path = path
            self.state.checkpoint_dataset = self.ckpt_dataset or ""
        return True

    def load_dataset(self, out_dir: str, dataset: str) -> bool:
        return self.load_checkpoint(checkpoint_path(out_dir, dataset), dataset)

    def load_latest(self, out_dir: str = "./checkpoints",
                    dataset: Optional[str] = None) -> bool:
        """Best-effort: prefer the named dataset, otherwise the most recently
        updated per-dataset checkpoint."""
        if dataset and self.load_dataset(out_dir, dataset):
            return True
        items = list_checkpoints(out_dir)
        if items:
            items.sort(key=lambda it: it["info"].get("updated_at", ""), reverse=True)
            return self.load_dataset(out_dir, items[0]["dataset"])
        return False

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
            ds_dir = dataset_dir(cfg.out_dir, cfg.dataset)
            os.makedirs(ds_dir, exist_ok=True)
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

            # Optional resume from checkpoints/<dataset>/latest.pt
            start_epoch = 0
            history: list = []
            if cfg.resume:
                ckpt_file = checkpoint_path(cfg.out_dir, cfg.dataset)
                if os.path.exists(ckpt_file):
                    try:
                        prev = torch.load(ckpt_file, map_location=device)
                        model.load_state_dict(prev["model"])
                        if "optimizer" in prev:
                            try:
                                opt.load_state_dict(prev["optimizer"])
                            except Exception:
                                pass
                        start_epoch = int(prev.get("meta", {}).get("epoch", 0))
                        info_file = info_path(cfg.out_dir, cfg.dataset)
                        if os.path.exists(info_file):
                            with open(info_file, "r", encoding="utf-8") as f:
                                history = json.load(f).get("history", []) or []
                        with self._lock:
                            self.state.message = f"resumed from epoch {start_epoch}"
                            self.state.history = list(history)
                    except Exception as e:
                        with self._lock:
                            self.state.message = f"resume failed: {e}"
                        start_epoch = 0
                        history = []
                else:
                    with self._lock:
                        self.state.message = "no prior checkpoint to resume"

            # Expose the live model so the web UI can query its tensor shapes
            # while training is in progress.
            self.model = model
            self.device = device

            steps_per_epoch = len(train_loader)
            # When resuming, treat cfg.epochs as "this many MORE epochs".
            target_epoch = start_epoch + cfg.epochs
            with self._lock:
                self.state.steps_per_epoch = steps_per_epoch
                self.state.total_epochs = target_epoch
                self.state.epoch = start_epoch
                self.state.message = f"training on {device}"

            for epoch in range(start_epoch + 1, target_epoch + 1):
                if self._stop.is_set():
                    break
                model.train()
                ep_loss = ep_recon = ep_kld = 0.0
                last_sample_t = 0.0  # wall-clock throttle for preview frames
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

                    # Throttle preview-image sampling to the UI-selected rate
                    # (Hz), independent of the actual training step rate.
                    now = time.monotonic()
                    fps = self._preview_fps
                    interval = 1.0 / max(1.0, min(30.0, fps))
                    if (now - last_sample_t) >= interval:
                        sample_png = _encode_sample_png(x_hat[0])
                        input_png  = _encode_sample_png(x[0])
                        last_sample_t = now
                    else:
                        sample_png = None
                        input_png  = None

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
                        if sample_png is not None:
                            self.state.sample_image = sample_png
                            self.state.input_image  = input_png

                n = max(1, step)
                entry = {"epoch": epoch,
                         "loss":  ep_loss  / n,
                         "recon": ep_recon / n,
                         "kld":   ep_kld   / n}
                history.append(entry)
                with self._lock:
                    self.state.history = list(history)
                    self.state.message = f"epoch {epoch} done"

                # save checkpoint each epoch under the per-dataset dir
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
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
                torch.save(ckpt, checkpoint_path(cfg.out_dir, cfg.dataset))
                info = _build_info(cfg, in_channels, image_size, history,
                                   epoch, entry["loss"], entry["recon"], entry["kld"],
                                   param_counts=model.parameter_counts())
                with open(info_path(cfg.out_dir, cfg.dataset), "w",
                          encoding="utf-8") as f:
                    json.dump(info, f, indent=2)
                self.ckpt_path = checkpoint_path(cfg.out_dir, cfg.dataset)
                self.ckpt_dataset = cfg.dataset
                self.ckpt_info = info
                with self._lock:
                    self.state.checkpoint_path = self.ckpt_path
                    self.state.checkpoint_dataset = cfg.dataset

            # auto-load the trained model for inference (refresh metadata)
            self.load_dataset(cfg.out_dir, cfg.dataset)
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
    p.add_argument("--hidden",  type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--device",  default="auto")
    p.add_argument("--resume",  action="store_true",
                   help="continue from checkpoints/<dataset>/latest.pt if present")
    args = p.parse_args()

    cfg = TrainConfig(dataset=args.dataset, data_root=args.data_root, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, beta=args.beta,
                      hidden=args.hidden, image_size=args.image_size,
                      num_workers=args.num_workers,
                      out_dir=args.out_dir, device=args.device,
                      resume=args.resume)

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
