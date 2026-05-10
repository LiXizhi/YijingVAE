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

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from train import TrainingManager, TrainConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
CORS(app)

manager = TrainingManager()
# best-effort load existing checkpoint at startup
manager.load_latest()


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
@app.post("/api/train/start")
def api_start():
    data = request.get_json(silent=True) or {}
    cfg = TrainConfig(
        dataset=data.get("dataset", "stl10"),
        epochs=int(data.get("epochs", 20)),
        batch_size=int(data.get("batch_size", 128)),
        lr=float(data.get("lr", 1e-3)),
        beta=float(data.get("beta", 4.0)),
        hidden=int(data.get("hidden", 64)),
        image_size=int(data.get("image_size", 64)),
        num_workers=int(data.get("num_workers", 0)),
        device=data.get("device", "cuda"),
    )
    ok = manager.start(cfg)
    return jsonify({"ok": ok, "state": manager.snapshot()})


@app.post("/api/train/stop")
def api_stop():
    ok = manager.stop()
    return jsonify({"ok": ok, "state": manager.snapshot()})


@app.get("/api/train/state")
def api_state():
    return jsonify(manager.snapshot())


# ---------- inference APIs ----------
@app.post("/api/model/load")
def api_load():
    ok = manager.load_latest()
    return jsonify({"ok": ok, "loaded": manager.model is not None})


@app.get("/api/model/info")
def api_info():
    return jsonify({"loaded": manager.model is not None,
                    "checkpoint": manager.ckpt_path})


@app.post("/api/decode")
def api_decode():
    """Decode a 6-D latent vector to an image and return as base64 PNG."""
    if manager.model is None:
        if not manager.load_latest():
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
