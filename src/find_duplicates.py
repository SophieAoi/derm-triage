"""Find exact and near-duplicate images in a dataset that has no patient or lesion IDs.

Three signals, each saved so thresholds can be changed without recomputing:
    file hash         byte-identical files
    perceptual hash   64-bit DCT hash (pHash); small Hamming distance = same photo resized,
                      recompressed or lightly edited
    CNN embedding     ImageNet ResNet-50 features; high cosine similarity = crops, zooms or
                      other shots of the same lesion

Duplicate rule (rule D, see DOCUMENTATION.md): a pair is a duplicate if
    cosine >= 0.92, or pHash distance <= 10 and cosine >= 0.80.
Duplicates are linked into groups (A~B and B~C puts A, B and C together) and each group
is kept in one split. No image is removed and no label is changed.

Outputs (in --out-dir, default data/<dataset>/duplicates/):
    features.npz                   file hashes, perceptual hashes and embeddings, in image_id order
    candidate_pairs.csv            every pair within the loose candidate limits, with both scores
    groups.csv                     image_id, group_id, group_size for every image (rule D)
    condition_label_conflicts.csv  near-certain duplicate pairs whose condition labels differ
    skin_type_conflicts.csv        near-certain duplicate pairs whose Fitzpatrick types differ

Usage:
    python src/find_duplicates.py                  # compute everything
    python src/find_duplicates.py --skip-features  # reuse candidate_pairs.csv, redo groups and lists
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
from src.train import pick_device  # noqa: E402

DATASETS = {
    # name: (metadata csv, image id column, image folder, file name pattern)
    "fitzpatrick": ("fitzpatrick/fitzpatrick17k.csv", "md5hash", "fitzpatrick/images", "{}.jpg"),
}
# Loose limits for writing candidate pairs; the duplicate rule itself is chosen afterwards.
CANDIDATE_MAX_HAMMING = 16
CANDIDATE_MIN_COSINE = 0.85

# Rule D, chosen after visual review of sampled pairs (see DOCUMENTATION.md).
DUPLICATE_MIN_COSINE = 0.92
DUPLICATE_MAX_HAMMING = 10
DUPLICATE_HAMMING_MIN_COSINE = 0.80  # stops pHash alone matching unrelated images

# Pairs treated as near-certainly the same image, for the label-conflict lists only.
CERTAIN_MAX_HAMMING = 4
CERTAIN_MIN_COSINE = 0.98


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


def compute_features(paths, device="auto"):
    file_hashes, phashes = [], []
    for n, path in enumerate(paths):
        file_hashes.append(file_hash(path))
        with Image.open(path) as img:
            phashes.append(perceptual_hash(img))
        if n % 2000 == 0:
            print(f"  hashes {n}/{len(paths)}", flush=True)
    return np.array(file_hashes), np.array(phashes, dtype=np.uint64), embeddings(paths, pick_device(device))


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


def is_duplicate(pairs):
    """Boolean mask of the candidate pairs that rule D counts as duplicates."""
    return (pairs["cosine"] >= DUPLICATE_MIN_COSINE) | (
        (pairs["phash_distance"] <= DUPLICATE_MAX_HAMMING) & (pairs["cosine"] >= DUPLICATE_HAMMING_MIN_COSINE))


def is_near_certain(pairs):
    return (pairs["phash_distance"] <= CERTAIN_MAX_HAMMING) | (pairs["cosine"] >= CERTAIN_MIN_COSINE)


def groups_table(ids, pairs):
    """One row per image: image_id, group_id, group_size, using rule D."""
    dup = pairs[is_duplicate(pairs)]
    groups = pd.Series(duplicate_groups(ids, zip(dup["image_a"], dup["image_b"])), name="group_id")
    table = groups.rename_axis("image_id").reset_index()
    table["group_size"] = table.groupby("group_id")["image_id"].transform("size")
    return table


def conflict_tables(pairs, metadata, id_column, label_column="label", skin_column="fitzpatrick_scale"):
    """Near-certain duplicate pairs whose condition labels, or skin types, differ."""
    meta = metadata.set_index(id_column)
    sure = pairs[is_near_certain(pairs)].reset_index(drop=True)
    for side in ("a", "b"):
        rows = meta.loc[sure[f"image_{side}"]]
        sure[f"label_{side}"] = rows[label_column].to_numpy()
        sure[f"fitzpatrick_{side}"] = rows[skin_column].to_numpy()
        sure[f"url_{side}"] = rows["url"].to_numpy() if "url" in rows else ""
    columns = ["image_a", "image_b", "phash_distance", "cosine", "label_a", "label_b",
               "fitzpatrick_a", "fitzpatrick_b", "url_a", "url_b"]
    sure = sure[columns]
    return sure[sure["label_a"] != sure["label_b"]], sure[sure["fitzpatrick_a"] != sure["fitzpatrick_b"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fitzpatrick")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--skip-features", action="store_true",
                        help="reuse an existing candidate_pairs.csv instead of recomputing")
    args = parser.parse_args()

    metadata_file, id_column, image_dir, pattern = DATASETS[args.dataset]
    out_dir = args.out_dir or args.data_dir / args.dataset / "duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.data_dir / metadata_file, dtype=str)
    ids = sorted(metadata[id_column])
    print(f"{len(ids)} images")

    if args.skip_features:
        pairs = pd.read_csv(out_dir / "candidate_pairs.csv", dtype={"image_a": str, "image_b": str})
    else:
        paths = [args.data_dir / image_dir / pattern.format(i) for i in ids]
        file_hashes, phashes, embeds = compute_features(paths, args.device)
        np.savez_compressed(out_dir / "features.npz", ids=np.array(ids), file_hashes=file_hashes,
                            phashes=phashes, embeddings=embeds.astype(np.float16))
        pairs = candidate_pairs(ids, phashes, embeds)
        pairs.to_csv(out_dir / "candidate_pairs.csv", index=False, float_format="%.5f")
    print(f"{len(pairs)} candidate pairs")

    groups = groups_table(ids, pairs)
    groups.to_csv(out_dir / "groups.csv", index=False)
    in_groups = groups[groups["group_size"] > 1]
    print(f"Rule D: {int(is_duplicate(pairs).sum())} duplicate pairs; {len(in_groups)} images in "
          f"{in_groups['group_id'].nunique()} groups; largest group {groups['group_size'].max()}")

    label_conflicts, skin_conflicts = conflict_tables(pairs, metadata, id_column)
    label_conflicts.to_csv(out_dir / "condition_label_conflicts.csv", index=False, float_format="%.5f")
    skin_conflicts.to_csv(out_dir / "skin_type_conflicts.csv", index=False, float_format="%.5f")
    print(f"Near-certain duplicate pairs: {int(is_near_certain(pairs).sum())}; "
          f"{len(label_conflicts)} with different condition labels, {len(skin_conflicts)} with different skin types")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
