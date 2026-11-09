import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/shakespeare_char"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "input.txt"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Tiny Shakespeare checksum mismatch: {actual_hash}")
    text = path.read_text()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    tokens = np.array([stoi[c] for c in text], dtype=np.uint16)
    # Contiguous splits prevent overlapping windows across train/val/test.
    n, m = int(0.9 * len(tokens)), int(0.95 * len(tokens))
    for name, chunk in zip(("train", "val", "test"), (tokens[:n], tokens[n:m], tokens[m:])):
        np.save(DATA / f"{name}.npy", chunk)
    meta = dict(source=URL, sha256=actual_hash,
                vocab=vocab, vocab_size=len(vocab), split_sizes=[n, m-n, len(tokens)-m])
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Batches:
    def __init__(self, batch_size, block_size, device):
        self.batch_size, self.block_size, self.device = batch_size, block_size, device
        self.arrays = {s: torch.from_numpy(np.load(DATA / f"{s}.npy").astype(np.int64)) for s in ("train", "val")}

    def get(self, split, generator):
        data = self.arrays[split]
        starts = torch.randint(len(data) - self.block_size, (self.batch_size,), generator=generator)
        x = torch.stack([data[i:i+self.block_size] for i in starts])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in starts])
        return x.to(self.device), y.to(self.device)


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
