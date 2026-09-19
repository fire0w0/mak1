"""Sparse connectome construction and a lightweight LIF reservoir.

Matrix convention: W[post, pre]. A presynaptic spike vector is multiplied as
W @ spikes, producing postsynaptic input. The real MaleCNS builder keeps the
graph sparse and incoming-normalizes absolute synaptic strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil
import tempfile
import numpy as np
from scipy import sparse

DATA = Path("data")
ARTIFACTS = Path("artifacts")
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA / "body-neurotransmitters-male-cns-v1.0.feather"


def _connectivity_batches():
    """Yield Feather record batches from a memory-mapped file.

    Unlike ``feather.read_table``, this never creates an Arrow table for the
    1+ GB connectivity file. A single record batch is the largest unit held in
    RAM, so peak memory is driven by the file's batch size rather than all 25M
    edges.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to build the MaleCNS matrix.") from exc
    source = pa.memory_map(str(WEIGHTS), "r")
    reader = ipc.open_file(source)
    try:
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)
    finally:
        source.close()


def build_male_cns(output: Path = ARTIFACTS / "male_cns_connectome.npz") -> tuple[int, int]:
    """Build a signed CSR matrix with two streaming passes over connectivity.

    The first pass measures per-postsynaptic normalization and CSR row sizes.
    The second fills on-disk memmaps directly at their final CSR offsets. This
    avoids a Python ID dictionary, a dense square matrix, and a full in-memory
    COO edge list.
    """
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to build the MaleCNS matrix.") from exc
    missing = [str(p) for p in (WEIGHTS, ANNOTATIONS, NEUROTRANSMITTERS) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing MaleCNS files:\n" + "\n".join(missing))
    annotations = feather.read_table(ANNOTATIONS).to_pandas()
    id_column = "bodyId" if "bodyId" in annotations else "body"
    neuron_ids = np.sort(annotations[id_column].dropna().drop_duplicates().to_numpy(dtype=np.int64))
    nt = feather.read_table(NEUROTRANSMITTERS, columns=["body", "consensus_nt"]).to_pandas().drop_duplicates("body").set_index("body")
    labels = nt.reindex(neuron_ids)["consensus_nt"].fillna("unclear").astype(str).str.lower()
    inhibitory = labels.str.contains("gaba|glutamate|histamine").to_numpy()
    signs = np.where(inhibitory, -1.0, 1.0).astype(np.float32)
    n = len(neuron_ids)
    incoming = np.zeros(n, dtype=np.float64)
    row_counts = np.zeros(n, dtype=np.int64)
    for number, batch in enumerate(_connectivity_batches(), start=1):
        pre_ids = batch["body_pre"].to_numpy(zero_copy_only=False)
        post_ids = batch["body_post"].to_numpy(zero_copy_only=False)
        values = batch["weight"].to_numpy(zero_copy_only=False).astype(np.float32)
        pre = np.searchsorted(neuron_ids, pre_ids)
        post = np.searchsorted(neuron_ids, post_ids)
        ok = (pre < n) & (post < n) & (neuron_ids[np.minimum(pre, n - 1)] == pre_ids) & (neuron_ids[np.minimum(post, n - 1)] == post_ids)
        valid_post, valid_values = post[ok], values[ok]
        incoming += np.bincount(valid_post, weights=np.abs(valid_values), minlength=n)
        row_counts += np.bincount(valid_post, minlength=n)
        print(f"Pass 1: record batch {number:,}; {int(row_counts.sum()):,} usable edges", flush=True)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    edge_count = int(indptr[-1])
    output.parent.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="flysnake-csr-", dir=output.parent))
    try:
        indices = np.memmap(temp_dir / "indices.bin", mode="w+", dtype=np.int32, shape=edge_count)
        data = np.memmap(temp_dir / "data.bin", mode="w+", dtype=np.float32, shape=edge_count)
        cursor = indptr[:-1].copy()
        for number, batch in enumerate(_connectivity_batches(), start=1):
            pre_ids = batch["body_pre"].to_numpy(zero_copy_only=False)
            post_ids = batch["body_post"].to_numpy(zero_copy_only=False)
            values = batch["weight"].to_numpy(zero_copy_only=False).astype(np.float32)
            pre, post = np.searchsorted(neuron_ids, pre_ids), np.searchsorted(neuron_ids, post_ids)
            ok = (pre < n) & (post < n) & (neuron_ids[np.minimum(pre, n - 1)] == pre_ids) & (neuron_ids[np.minimum(post, n - 1)] == post_ids)
            pre, post, values = pre[ok].astype(np.int32), post[ok].astype(np.int32), values[ok]
            # Connectivity contains segments outside the curated annotation
            # index. Some Arrow batches can therefore have no usable edges.
            if post.size == 0:
                print(f"Pass 2: record batch {number:,}; no annotated edges", flush=True)
                continue
            values = values * signs[pre] / np.maximum(incoming[post], 1.0)
            # Group inside this one batch so each CSR row receives a contiguous write.
            order = np.argsort(post, kind="stable")
            pre, post, values = pre[order], post[order], values[order]
            starts = np.r_[0, np.flatnonzero(np.diff(post)) + 1]
            stops = np.r_[starts[1:], len(post)]
            for start, stop in zip(starts, stops):
                row = int(post[start]); destination = int(cursor[row]); count = int(stop - start)
                indices[destination : destination + count] = pre[start:stop]
                data[destination : destination + count] = values[start:stop]
                cursor[row] += count
            print(f"Pass 2: record batch {number:,}", flush=True)
        indices.flush(); data.flush()
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n, n), dtype=np.float32, copy=False)
        # MaleCNS flat connectivity is expected to have one row per body pair;
        # if future releases contain duplicates, canonicalize before persisting.
        matrix.sum_duplicates()
        sparse.save_npz(output, matrix, compressed=True)
        np.save(output.with_suffix(".neuron_ids.npy"), neuron_ids)
        del matrix
    finally:
        # Drop references before Windows removes the memory-mapped staging files.
        try:
            del data, indices
        except UnboundLocalError:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)
    return n, edge_count


def synthetic_connectome(neurons: int = 420, degree: int = 10, seed: int = 9) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    post = np.repeat(np.arange(neurons), degree)
    pre = rng.integers(0, neurons, size=len(post))
    values = rng.gamma(1.5, 0.6, len(post)).astype(np.float32)
    values *= rng.choice(np.array([-1.0, 1.0], dtype=np.float32), len(post), p=(0.23, 0.77))
    matrix = sparse.csr_matrix((values, (post, pre)), shape=(neurons, neurons), dtype=np.float32)
    normalizer = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    return sparse.diags(1 / np.maximum(normalizer, 1.0)) @ matrix


@dataclass
class LIFReservoir:
    W: sparse.csr_matrix
    dt_ms: float = 1.0
    v_rest: float = -52.0
    threshold: float = -45.0
    reset: float = -52.0
    tau_m: float = 20.0
    refractory_ms: float = 2.0
    synaptic_scale: float = 4.0
    external_scale: float = 8.0
    seed: int = 11

    def response(self, external_rates: np.ndarray, duration_ms: int = 70) -> tuple[np.ndarray, np.ndarray]:
        """Run an LIF trial; return spike counts and active-neuron indices."""
        rng = np.random.default_rng(self.seed + int(external_rates.sum() * 100))
        n = self.W.shape[0]
        voltage = np.full(n, self.v_rest, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.int16)
        previous = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        for _ in range(duration_ms):
            external = rng.random(n) < external_rates * (self.dt_ms / 1000.0)
            recurrent = self.W @ previous
            active = refractory == 0
            voltage[active] += ((self.v_rest - voltage[active]) / self.tau_m + self.synaptic_scale * recurrent[active] + external[active] * self.external_scale) * self.dt_ms
            spikes = (voltage >= self.threshold) & active
            voltage[spikes] = self.reset
            refractory = np.maximum(refractory - 1, 0)
            refractory[spikes] = int(self.refractory_ms / self.dt_ms)
            previous = spikes.astype(np.float32)
            counts += previous
        return counts / (duration_ms / 1000.0), np.flatnonzero(counts)


def save_matrix(matrix: sparse.csr_matrix, neuron_ids: np.ndarray, path: Path = ARTIFACTS / "male_cns_connectome.npz") -> None:
    path.parent.mkdir(exist_ok=True)
    sparse.save_npz(path, matrix)
    np.save(path.with_suffix(".neuron_ids.npy"), neuron_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-male-cns", action="store_true")
    args = parser.parse_args()
    if args.build_male_cns:
        n, edges = build_male_cns()
        print(f"Saved {n:,} × {n:,} sparse matrix with {edges:,} source edges.")
    else:
        W = synthetic_connectome(); print(f"Synthetic reservoir: {W.shape}, {W.nnz:,} edges")
