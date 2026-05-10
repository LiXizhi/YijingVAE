# Yijing β-VAE

A minimal training framework for a **β-VAE with 6 latent variables**, one for
each *yao* (爻) of an *I Ching* hexagram — plus two HTML5 pages: one to
**train** the model from your browser and one to **play** with the six lines
and watch the model imagine images.

```
YijingVAE/
├── model.py        # BetaVAE (6-dim latent) + loss
├── datasets.py     # MNIST / FashionMNIST loaders
├── train.py        # CLI training + threaded TrainingManager
├── server.py       # Flask server (UI + JSON API)
├── static/
│   ├── index.html  # launcher
│   ├── train.html  # live training dashboard
│   └── play.html   # 6-yao hexagram explorer
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch may need a CUDA-specific install — see https://pytorch.org if you
want GPU support. CPU works fine for MNIST.

## Run the web app

```powershell
python server.py
```

Then open:

- http://localhost:5000/         — launcher
- http://localhost:5000/train    — start/stop training, live loss chart
- http://localhost:5000/play     — six sliders = six yao; the model decodes them

Training writes checkpoints to `./checkpoints/latest.pt` after every epoch;
the play page automatically uses the latest checkpoint.

## Train from the CLI instead

```powershell
python train.py --dataset mnist --epochs 20 --beta 4.0
```

Arguments: `--dataset {mnist,fashion}`, `--epochs`, `--batch-size`, `--lr`,
`--beta`, `--hidden`, `--device {auto,cpu,cuda}`, `--out-dir`.

## Notes on the six-yao mapping

The β-VAE encourages the 6 latent dimensions to be approximately independent
factors of variation. In the play page, the **sign** of each latent is shown
as a *yang* (solid) or *yin* (broken) line, and the **magnitude** controls
how strongly that factor is expressed. Hexagrams are read bottom-up, matching
classical *Yijing* convention.
