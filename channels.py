"""Translate a local Snake view into six artificial sensory channels."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

CHANNELS = ("food_left", "food_straight", "food_right", "danger_left", "danger_straight", "danger_right")
ACTIONS = ("left", "straight", "right")


@dataclass(frozen=True)
class ChannelMap:
    input_indices: np.ndarray
    readout_indices: np.ndarray


def make_channel_map(neuron_count: int, seed: int = 7, inputs_per_channel: int = 12, readout_count: int = 96) -> ChannelMap:
    """Choose reproducible stand-ins until biological annotations are curated.

    The readout deliberately includes the stimulated sensory neurons, plus a
    random set of downstream candidates. This lets the linear policy learn a
    useful controller before a biologically curated DN list is available.
    """
    if neuron_count < len(CHANNELS) * inputs_per_channel + readout_count:
        raise ValueError("reservoir is too small for the configured channel map")
    rng = np.random.default_rng(seed)
    sensory_count = len(CHANNELS) * inputs_per_channel
    if readout_count < sensory_count:
        raise ValueError("readout_count must include all sensory targets")
    picks = rng.choice(neuron_count, sensory_count + (readout_count - sensory_count), replace=False)
    inputs = picks[:sensory_count].reshape(len(CHANNELS), inputs_per_channel)
    return ChannelMap(inputs, picks)


def encode(food_relation: int, danger: tuple[bool, bool, bool], active_rate: float = 150.0) -> np.ndarray:
    """Return rates ordered left/straight/right for food, then danger.

    `food_relation` is -1, 0, or +1 relative to the snake's heading.
    """
    rates = np.zeros(len(CHANNELS), dtype=np.float32)
    rates[food_relation + 1] = active_rate
    rates[3:] = np.asarray(danger, dtype=np.float32) * active_rate
    return rates


def inject_rates(rates: np.ndarray, channel_map: ChannelMap, neuron_count: int) -> np.ndarray:
    """Spread each channel rate over its selected sensory-neuron stand-ins."""
    external = np.zeros(neuron_count, dtype=np.float32)
    for channel, targets in enumerate(channel_map.input_indices):
        external[targets] += rates[channel] / len(targets)
    return external
