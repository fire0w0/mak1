"""Precompute reservoir responses for all simple sensory situations."""

from pathlib import Path
import numpy as np

from brain import LIFReservoir, synthetic_connectome
from channels import CHANNELS, encode, inject_rates, make_channel_map

OUT = Path("artifacts/response_bank.npz")


def main() -> None:
    W = synthetic_connectome()
    reservoir = LIFReservoir(W)
    mapping = make_channel_map(W.shape[0])
    features, labels = [], []
    # Relation x danger pattern, repeated so the classifier learns robustly.
    for relation in (-1, 0, 1):
        for mask in range(8):
            danger = tuple(bool(mask & (1 << bit)) for bit in range(3))
            action = relation + 1 if not danger[relation + 1] else next((a for a in (1, 0, 2) if not danger[a]), 1)
            for repeat in range(14):
                rates = encode(relation, danger)
                response, _ = reservoir.response(inject_rates(rates, mapping, W.shape[0]), duration_ms=70 + repeat % 4)
                features.append(response[mapping.readout_indices]); labels.append(action)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, X=np.asarray(features, dtype=np.float32), y=np.asarray(labels), readout_indices=mapping.readout_indices)
    print(f"Wrote {len(features)} response samples to {OUT}")


if __name__ == "__main__":
    main()
