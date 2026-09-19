"""Local live demo API for the synthetic FlySnake MVP.

The real MaleCNS builder is separate by design: it is an offline preparation
step, while this process serves low-latency gameplay from a small reservoir.
"""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import numpy as np

from brain import LIFReservoir, synthetic_connectome
from channels import encode, inject_rates, make_channel_map
from snake import SnakeGame

ROOT = Path(__file__).parent
MODEL = ROOT / "artifacts" / "readout.npz"
game = SnakeGame()
W = synthetic_connectome()
reservoir = LIFReservoir(W)
channel_map = make_channel_map(W.shape[0])
model = None


def load_model() -> dict:
    global model
    if model is None:
        if not MODEL.exists():
            raise RuntimeError("Run `python build_bank.py` then `python train.py` before starting the demo.")
        model = dict(np.load(MODEL))
    return model


def play_step() -> dict:
    parameters = load_model()
    relation, danger = game.sensory_state()
    rates = encode(relation, danger)
    response, active = reservoir.response(inject_rates(rates, channel_map, W.shape[0]))
    feature = (response[parameters["readout_indices"]] - parameters["mean"]) / parameters["scale"]
    logits = feature @ parameters["weights"] + parameters["bias"]
    probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
    action = int(probabilities.argmax())
    state = game.step(action)
    return {**state, "action": ("left", "straight", "right")[action], "probabilities": probabilities.round(4).tolist(), "active_neurons": active[:150].tolist(), "stimulus": dict(zip(("food_left", "food_straight", "food_right", "danger_left", "danger_straight", "danger_right"), rates.tolist()))}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/step":
                payload = play_step()
            elif self.path == "/api/reset":
                game.reset(); payload = game.snapshot()
            else:
                self.send_error(404); return
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except RuntimeError as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(409); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    print("FlySnake demo: http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
