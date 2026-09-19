# FlySnake

FlySnake is a small, runnable reservoir-computing demo: a fixed fly-connectome-inspired spiking network receives Snake state, and a trainable linear readout chooses **left**, **straight**, or **right**.

It runs immediately with a deterministic synthetic connectome. When the three MaleCNS v1.0 Feather files are placed in `data/`, `brain.py` can build the signed, incoming-normalized sparse connectivity matrix used by the same simulation.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python build_bank.py
python train.py
python server.py
```

Open `http://127.0.0.1:8000`. The page runs the trained controller and visualizes its active reservoir neurons and action probabilities.

## Real MaleCNS data (optional)

Put these files in `data/`:

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`

Then run `python brain.py --build-male-cns`. The builder expects `body_pre`, `body_post`, and `weight` in connectivity data, and uses presynaptic `consensus_nt` to assign signs. It produces `artifacts/male_cns_connectome.npz`; it deliberately does not load the entire graph into a dense matrix.

### Large-data behaviour

The connectivity builder is intentionally a two-pass, memory-mapped pipeline.

1. It memory-maps the 1+ GB Feather file and streams one Arrow record batch at a time.
2. Pass 1 maps body IDs using `numpy.searchsorted`, counts CSR rows, and computes each postsynaptic neuron's incoming absolute-weight normalizer.
3. Pass 2 writes normalized signed edges directly into disk-backed CSR staging arrays, then saves a compressed SciPy CSR artifact and the matching body-ID vector.

This keeps the representation sparse (`W[post, pre]`) and avoids both a Python dictionary over 166k neuron IDs and a dense 166,700 × 166,700 allocation. Reserve several GB of free disk for the transient CSR staging files and output artifact. The command logs its record-batch progress so it is safe to monitor in a terminal; it never downloads the dataset automatically.

The demo's synthetic fallback is for development and presentation only. Real-data runs need resource planning: the full graph contains millions of edges and should be built and benchmarked before enabling live gameplay.

## Project layout

- `brain.py` — sparse graph builders and leaky-integrate-and-fire reservoir
- `channels.py` — Snake-to-sensory-rate encoder and output feature selection
- `snake.py` — deterministic Snake environment and safe teacher policy
- `build_bank.py` — produces repeated brain responses for every sensory state
- `train.py` — trains the linear readout by imitation learning
- `server.py` — local JSON API and browser demo
