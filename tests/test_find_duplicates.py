"""Tests for duplicate detection. Run from the repo root: python3 -m pytest tests"""

import unittest

import numpy as np
from PIL import Image, ImageFilter

import pandas as pd

from src.find_duplicates import (
    candidate_pairs,
    conflict_tables,
    duplicate_groups,
    groups_table,
    hamming_matrix,
    is_duplicate,
    perceptual_hash,
)


def textured_image(seed, size=(300, 220)):
    """A smooth random pattern, so resizing keeps its structure (as with real photos)."""
    rng = np.random.default_rng(seed)
    small = (rng.random((12, 16, 3)) * 255).astype(np.uint8)
    return Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)


class PerceptualHashTests(unittest.TestCase):
    def distance(self, a, b):
        return int(hamming_matrix(np.array([perceptual_hash(a)]), np.array([perceptual_hash(b)]))[0, 0])

    def test_identical_image_has_distance_zero(self):
        img = textured_image(0)
        self.assertEqual(self.distance(img, img.copy()), 0)

    def test_resized_and_blurred_copy_stays_close(self):
        img = textured_image(1)
        edited = img.resize((150, 110)).filter(ImageFilter.GaussianBlur(1))
        self.assertLessEqual(self.distance(img, edited), 6)

    def test_different_images_are_far_apart(self):
        distances = [self.distance(textured_image(s), textured_image(s + 100)) for s in range(10)]
        self.assertGreater(min(distances), 12)


class HammingTests(unittest.TestCase):
    def test_counts_differing_bits(self):
        a = np.array([0b0000, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        b = np.array([0b1011], dtype=np.uint64)
        np.testing.assert_array_equal(hamming_matrix(a, b), [[3], [61]])


class CandidatePairTests(unittest.TestCase):
    def test_pairs_found_by_either_signal_once_each(self):
        ids = ["a", "b", "c", "d"]
        phashes = np.array([0, 1, 0xFFFFFFFFFFFFFFFF, 0xFFFF0000FFFF0000], dtype=np.uint64)
        embeds = np.eye(4)
        embeds[3] = embeds[2]  # c and d: same embedding, very different hashes
        pairs = candidate_pairs(ids, phashes, embeds, max_hamming=2, min_cosine=0.9, chunk=3)
        found = {(r.image_a, r.image_b) for r in pairs.itertuples()}
        self.assertEqual(found, {("a", "b"), ("c", "d")})


class GroupTests(unittest.TestCase):
    def test_chains_join_one_group(self):
        groups = duplicate_groups(["a", "b", "c", "d", "e"], [("c", "b"), ("b", "a"), ("e", "d")])
        self.assertEqual(groups, {"a": "a", "b": "a", "c": "a", "d": "d", "e": "d"})

    def test_pair_order_does_not_matter(self):
        ids = [f"img{i}" for i in range(20)]
        pairs = [("img3", "img7"), ("img7", "img15"), ("img1", "img2"), ("img15", "img9")]
        first = duplicate_groups(ids, pairs)
        second = duplicate_groups(list(reversed(ids)), [(b, a) for a, b in reversed(pairs)])
        self.assertEqual(first, second)

    def test_unpaired_images_are_their_own_group(self):
        self.assertEqual(duplicate_groups(["x", "y"], []), {"x": "x", "y": "y"})


def pairs_frame(rows):
    return pd.DataFrame(rows, columns=["image_a", "image_b", "phash_distance", "cosine"])


class RuleDTests(unittest.TestCase):
    def test_rule_boundaries(self):
        pairs = pairs_frame([
            ("a", "b", 30, 0.92),   # cosine at the limit: duplicate
            ("a", "c", 30, 0.919),  # just below, hash far: not
            ("a", "d", 10, 0.80),   # hash at the limit with cosine floor met: duplicate
            ("a", "e", 10, 0.799),  # hash close but cosine below the floor: not
            ("a", "f", 12, 0.90),   # hash just too far, cosine too low: not
        ])
        self.assertEqual(is_duplicate(pairs).tolist(), [True, False, True, False, False])

    def test_groups_table_has_every_image_and_group_sizes(self):
        pairs = pairs_frame([("a", "b", 2, 0.95), ("b", "c", 2, 0.95), ("d", "e", 30, 0.5)])
        table = groups_table(["a", "b", "c", "d", "e"], pairs).set_index("image_id")
        self.assertEqual(table["group_id"].to_dict(), {"a": "a", "b": "a", "c": "a", "d": "d", "e": "e"})
        self.assertEqual(table["group_size"].to_dict(), {"a": 3, "b": 3, "c": 3, "d": 1, "e": 1})


class ConflictTests(unittest.TestCase):
    def test_only_near_certain_pairs_with_differences_are_listed(self):
        metadata = pd.DataFrame({
            "md5hash": ["a", "b", "c", "d"],
            "label": ["melanoma", "lentigo maligna", "psoriasis", "psoriasis"],
            "fitzpatrick_scale": ["2", "2", "3", "5"],
            "url": ["u1", "u2", "u3", "u4"],
        })
        pairs = pairs_frame([
            ("a", "b", 2, 0.99),   # near-certain, labels differ
            ("c", "d", 0, 0.99),   # near-certain, skin types differ
            ("a", "c", 12, 0.93),  # duplicate under rule D but not near-certain: not listed
        ])
        labels, skin = conflict_tables(pairs, metadata, "md5hash")
        self.assertEqual(list(zip(labels["image_a"], labels["image_b"])), [("a", "b")])
        self.assertEqual(list(zip(skin["image_a"], skin["image_b"])), [("c", "d")])
        self.assertEqual(skin.iloc[0][["fitzpatrick_a", "fitzpatrick_b"]].tolist(), ["3", "5"])


if __name__ == "__main__":
    unittest.main()
