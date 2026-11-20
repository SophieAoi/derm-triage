"""Find exact and near-duplicate images in a dataset that has no patient or lesion IDs.

Three signals, each saved so thresholds can be changed without recomputing:
    file hash         byte-identical files
    perceptual hash   64-bit DCT hash (pHash); small Hamming distance = same photo resized,
                      recompressed or lightly edited
    CNN embedding     ImageNet ResNet-50 features; high cosine similarity = crops, zooms or
                      other shots of the same lesion

Outputs (in --out-dir, default data/<dataset>/duplicates/):
    features.npz       file hashes, perceptual hashes and embeddings, in image_id order
    candidate_pairs.csv every pair within the loose candidate limits below, with both scores

Usage:
    python src/find_duplicates.py
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DATA_DIR, eval_transform  # noqa: E402
from src.model import build_model  # noqa: E402

DATASETS = {
    # name: (metadata csv, image id column, image folder, file name pattern)
    "fitzpatrick": ("fitzpatrick/fitzpatrick17k.csv", "md5hash", "fitzpatrick/images", "{}.jpg"),
}
# Loose limits for writing candidate pairs; the duplicate rule itself is chosen afterwards.
CANDIDATE_MAX_HAMMING = 16
CANDIDATE_MIN_COSINE = 0.85


def file_hash(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def perceptual_hash(image):
    """64-bit pHash: greyscale 32x32, 2-D DCT, top-left 8x8 low frequencies (minus DC) vs their median."""
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(pixels, norm="ortho")[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return np.packbits(bits).view(">u8")[0]


def hamming_matrix(hashes_a, hashes_b):
    """Pairwise Hamming distances between two arrays of 64-bit hashes."""
    return np.bitwise_count(np.bitwise_xor(hashes_a[:, None], hashes_b[None, :])).astype(np.uint8)


@torch.no_grad()
def embeddings(paths, device="cpu", batch_size=64):
    """L2-normalised ImageNet ResNet-50 features (the 2048-d layer before the classifier)."""
    model = build_model(1000, "resnet50", pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    transform = eval_transform()
    out = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as img:
                batch.append(transform(img.convert("RGB")))
        features = model(torch.stack(batch).to(device))
        out.append(torch.nn.functional.normalize(features, dim=1).cpu())
        if (start // batch_size) % 20 == 0:
            print(f"  embeddings {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(out).numpy()


def compute_features(paths):
    file_hashes, phashes = [], []
    for n, path in enumerate(paths):
        file_hashes.append(file_hash(path))
        with Image.open(path) as img:
            phashes.append(perceptual_hash(img))
        if n % 2000 == 0:
            print(f"  hashes {n}/{len(paths)}", flush=True)
    return np.array(file_hashes), np.array(phashes, dtype=np.uint64), embeddings(paths)


def candidate_pairs(ids, phashes, embeds, max_hamming=CANDIDATE_MAX_HAMMING,
                    min_cosine=CANDIDATE_MIN_COSINE, chunk=1000):
    """Every pair (i < j) with Hamming distance <= max_hamming or cosine >= min_cosine."""
    rows = []
    n = len(ids)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        ham = hamming_matrix(phashes[start:stop], phashes)
        cos = embeds[start:stop] @ embeds.T
        hit = (ham <= max_hamming) | (cos >= min_cosine)
        for a, b in zip(*np.nonzero(hit)):
            i, j = start + a, b
            if i < j:
                rows.append((ids[i], ids[j], int(ham[a, b]), float(cos[a, b])))
    return pd.DataFrame(rows, columns=["image_a", "image_b", "phash_distance", "cosine"])


def duplicate_groups(ids, pairs):
    """Group IDs so that any two images linked by a chain of pairs share a group.

    Returns {image_id: group_id}; the group ID is the alphabetically first image in the
    group, so it does not depend on pair order. Images with no pairs are their own group.
    """
    parent = {i: i for i in ids}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: root(i) for i in ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fitzpatrick")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    metadata, id_column, image_dir, pattern = DATASETS[args.dataset]
    out_dir = args.out_dir or args.data_dir / args.dataset / "duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(pd.read_csv(args.data_dir / metadata, dtype=str)[id_column])
    paths = [args.data_dir / image_dir / pattern.format(i) for i in ids]

    print(f"{len(ids)} images")
    file_hashes, phashes, embeds = compute_features(paths)
    np.savez_compressed(out_dir / "features.npz", ids=np.array(ids), file_hashes=file_hashes,
                        phashes=phashes, embeddings=embeds.astype(np.float16))
    pairs = candidate_pairs(ids, phashes, embeds)
    pairs.to_csv(out_dir / "candidate_pairs.csv", index=False, float_format="%.5f")
    print(f"{len(pairs)} candidate pairs saved to {out_dir}")


if __name__ == "__main__":
    main()
