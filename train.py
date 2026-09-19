"""Train the only plastic component: a linear action readout."""

from pathlib import Path
import numpy as np

BANK = Path("artifacts/response_bank.npz")
MODEL = Path("artifacts/readout.npz")


def main() -> None:
    if not BANK.exists():
        raise FileNotFoundError("Run `py build_bank.py` first.")
    bank = np.load(BANK)
    X, y = bank["X"], bank["y"]
    mean, scale = X.mean(0), X.std(0) + 1e-5
    X = (X - mean) / scale
    weights = np.zeros((X.shape[1], 3), dtype=np.float32); bias = np.zeros(3, dtype=np.float32)
    target = np.eye(3, dtype=np.float32)[y]
    for _ in range(900):
        logits = X @ weights + bias
        probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
        error = (probs - target) / len(X)
        weights -= 0.18 * (X.T @ error + 1e-4 * weights)
        bias -= 0.18 * error.sum(0)
    accuracy = ((X @ weights + bias).argmax(1) == y).mean()
    MODEL.parent.mkdir(exist_ok=True)
    np.savez(MODEL, weights=weights, bias=bias, mean=mean, scale=scale, readout_indices=bank["readout_indices"])
    print(f"Saved linear readout ({accuracy:.1%} bank accuracy) to {MODEL}")


if __name__ == "__main__":
    main()
