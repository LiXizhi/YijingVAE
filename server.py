"""Flask server: serves the training dashboard, the play UI, and JSON APIs.

Run:
    python server.py
Then open:
    http://localhost:5000/        -> launcher
    http://localhost:5000/train   -> training dashboard
    http://localhost:5000/play    -> hexagram explorer
"""

from __future__ import annotations

import base64
import io
import os
import threading

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from datasets import dataset_available, get_dataloaders
from train import TrainingManager, TrainConfig


def _device_info(device=None) -> dict:
    """Describe the compute device currently in use (or that would be used)."""
    if device is not None:
        kind = device.type
    elif torch.cuda.is_available():
        kind = "cuda"
    else:
        kind = "cpu"
    if kind == "cuda":
        try:
            idx = device.index if (device is not None and device.index is not None) else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory
            return {"kind": "cuda", "name": name, "index": idx,
                    "memory_gb": round(mem / (1024 ** 3), 1),
                    "label": f"GPU · {name} ({round(mem / (1024 ** 3), 1)} GB)"}
        except Exception as e:
            return {"kind": "cuda", "name": "GPU", "label": f"GPU ({e})"}
    cores = os.cpu_count() or 1
    threads = torch.get_num_threads()
    return {"kind": "cpu", "cores": cores, "threads": threads,
            "label": f"CPU · {cores} 核 / {threads} 线程"}

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
DATA_ROOT = os.path.join(ROOT, "data")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
SUPPORTED_DATASETS = {
    "stl10": {"label": "STL-10", "image_size": 64, "batch_size": 128},
    "cifar10": {"label": "CIFAR-10", "image_size": 64, "batch_size": 128},
    "mnist": {"label": "MNIST", "image_size": 28, "batch_size": 128},
    "fashion": {"label": "FashionMNIST", "image_size": 28, "batch_size": 128},
}

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
CORS(app)

manager = TrainingManager()
# best-effort load existing checkpoint at startup
manager.load_latest(CHECKPOINTS)


def _build_arch(model):
    """Return a JSON-friendly description of the tensor shapes flowing
    through the VAE for a single input. Used by the web UI to visualize
    the network during both training and inference."""
    if model is None:
        return None
    c, s, h = model.in_channels, model.image_size, model.hidden
    enc = model.enc_size
    feat = model._feat
    shapes = [
        {"stage": "输入图像",        "shape": [c, s, s],                  "note": "原始输入 (C, H, W)"},
        {"stage": "尺寸对齐",        "shape": [c, enc, enc],              "note": "缩放到 8 的倍数"},
        {"stage": "编码器 Conv1",    "shape": [h, enc // 2, enc // 2],     "note": "stride 2"},
        {"stage": "编码器 Conv2",    "shape": [h * 2, enc // 4, enc // 4], "note": "stride 2"},
        {"stage": "编码器 Conv3",    "shape": [h * 4, feat, feat],         "note": "stride 2"},
        {"stage": "展平",            "shape": [h * 4 * feat * feat],       "note": "Flatten"},
        {"stage": "潜变量 μ / logσ²", "shape": [model.latent_dim],         "note": "线性投影到 6 维"},
        {"stage": "六爻潜向量 z",     "shape": [model.latent_dim],         "note": "重参数化采样 (六爻)"},
        {"stage": "解码器 fc",       "shape": [h * 4, feat, feat],         "note": "线性 + reshape"},
        {"stage": "解码器 Deconv1",  "shape": [h * 2, enc // 4, enc // 4], "note": "stride 2"},
        {"stage": "解码器 Deconv2",  "shape": [h, enc // 2, enc // 2],     "note": "stride 2"},
        {"stage": "解码器 Deconv3",  "shape": [c, enc, enc],              "note": "stride 2 + sigmoid"},
        {"stage": "输出图像",        "shape": [c, s, s],                  "note": "缩放回原始尺寸"},
    ]
    return {
        "in_channels": c, "image_size": s, "hidden": h,
        "enc_size": enc, "feat": feat, "latent_dim": model.latent_dim,
        "shapes": shapes,
    }

download_lock = threading.Lock()
download_state = {
    "running": False,
    "dataset": None,
    "message": "idle",
    "ok": None,
    "available": False,
}


def _validate_dataset(name: str):
    dataset = str(name or "stl10").lower()
    if dataset not in SUPPORTED_DATASETS:
        return None, jsonify({
            "ok": False,
            "error": f"unsupported dataset: {dataset}",
            "datasets": list(SUPPORTED_DATASETS),
        }), 400
    return dataset, None, None


def _dataset_info(name: str) -> dict:
    info = dict(SUPPORTED_DATASETS[name])
    info["available"] = dataset_available(name, DATA_ROOT)
    return info


def _download_dataset(name: str):
    try:
        with download_lock:
            download_state.update({
                "running": True,
                "dataset": name,
                "message": f"downloading {name}",
                "ok": None,
                "available": dataset_available(name, DATA_ROOT),
            })

        defaults = SUPPORTED_DATASETS[name]
        get_dataloaders(
            name,
            data_root=DATA_ROOT,
            batch_size=defaults["batch_size"],
            image_size=defaults["image_size"],
        )
        available = dataset_available(name, DATA_ROOT)
        with download_lock:
            download_state.update({
                "running": False,
                "message": "downloaded" if available else "download incomplete",
                "ok": available,
                "available": available,
            })
    except Exception as e:
        with download_lock:
            download_state.update({
                "running": False,
                "message": f"error: {e}",
                "ok": False,
                "available": dataset_available(name, DATA_ROOT),
            })


# ---------- pages ----------
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/train")
def train_page():
    return send_from_directory(STATIC, "train.html")


@app.route("/play")
def play_page():
    return send_from_directory(STATIC, "play.html")


# ---------- training APIs ----------
@app.get("/api/datasets")
def api_datasets():
    return jsonify({"datasets": {name: _dataset_info(name) for name in SUPPORTED_DATASETS}})


@app.get("/api/datasets/<name>")
def api_dataset_status(name):
    dataset, error, status = _validate_dataset(name)
    if error is not None:
        return error, status
    return jsonify({"ok": True, "dataset": dataset, "info": _dataset_info(dataset)})


@app.post("/api/datasets/download")
def api_dataset_download():
    data = request.get_json(silent=True) or {}
    dataset, error, status = _validate_dataset(data.get("dataset", "stl10"))
    if error is not None:
        return error, status

    with download_lock:
        if download_state["running"]:
            return jsonify({"ok": False, "state": dict(download_state), "error": "download already running"}), 409
        download_state.update({
            "running": True,
            "dataset": dataset,
            "message": "starting",
            "ok": None,
            "available": dataset_available(dataset, DATA_ROOT),
        })

    thread = threading.Thread(target=_download_dataset, args=(dataset,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "state": dict(download_state)})


@app.get("/api/datasets/download/state")
def api_dataset_download_state():
    with download_lock:
        state = dict(download_state)
    dataset = state.get("dataset")
    if dataset in SUPPORTED_DATASETS:
        state["available"] = dataset_available(dataset, DATA_ROOT)
    return jsonify(state)


@app.post("/api/train/start")
def api_start():
    data = request.get_json(silent=True) or {}
    dataset, error, status = _validate_dataset(data.get("dataset", "stl10"))
    if error is not None:
        return error, status

    dataset_defaults = SUPPORTED_DATASETS[dataset]
    cfg = TrainConfig(
        dataset=dataset,
        data_root=data.get("data_root", DATA_ROOT),
        epochs=int(data.get("epochs", 20)),
        batch_size=int(data.get("batch_size", dataset_defaults["batch_size"])),
        lr=float(data.get("lr", 1e-3)),
        beta=float(data.get("beta", 4.0)),
        hidden=int(data.get("hidden", 64)),
        image_size=int(data.get("image_size", dataset_defaults["image_size"])),
        num_workers=int(data.get("num_workers", 0)),
        out_dir=data.get("out_dir", CHECKPOINTS),
        device=data.get("device", "auto"),
    )
    ok = manager.start(cfg)
    return jsonify({"ok": ok, "state": manager.snapshot()})


@app.post("/api/train/stop")
def api_stop():
    ok = manager.stop()
    return jsonify({"ok": ok, "state": manager.snapshot()})


@app.get("/api/train/state")
def api_state():
    snap = manager.snapshot()
    arch = _build_arch(manager.model)
    if arch is not None:
        snap["arch"] = arch
    snap["device_info"] = _device_info(manager.device)
    return jsonify(snap)


# ---------- inference APIs ----------
@app.post("/api/model/load")
def api_load():
    ok = manager.load_latest(CHECKPOINTS)
    return jsonify({"ok": ok, "loaded": manager.model is not None})


@app.get("/api/model/info")
def api_info():
    info = {"loaded": manager.model is not None,
            "checkpoint": manager.ckpt_path,
            "device_info": _device_info(manager.device)}
    arch = _build_arch(manager.model)
    if arch is not None:
        info["arch"] = arch
    return jsonify(info)


@app.post("/api/decode")
def api_decode():
    """Decode a 6-D latent vector to an image and return as base64 PNG."""
    if manager.model is None:
        if not manager.load_latest(CHECKPOINTS):
            return jsonify({"ok": False, "error": "no checkpoint available"}), 400

    data = request.get_json(silent=True) or {}
    z = data.get("z")
    if not isinstance(z, list) or len(z) != 6:
        return jsonify({"ok": False, "error": "z must be a list of 6 floats"}), 400
    try:
        z = [float(v) for v in z]
    except Exception:
        return jsonify({"ok": False, "error": "z values must be numeric"}), 400

    arr = manager.decode(z)  # (C,H,W) in [0,1]
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if img.shape[0] == 1:
        pil = Image.fromarray(img[0], mode="L")
    else:
        pil = Image.fromarray(np.transpose(img, (1, 2, 0)), mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({"ok": True,
                    "image": f"data:image/png;base64,{b64}",
                    "z": z})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
