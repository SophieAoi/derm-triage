"""Tests for duplicate detection. Run from the repo root: python3 -m pytest tests"""

import unittest

import numpy as np
from PIL import Image, ImageFilter

from src.find_duplicates import candidate_pairs, duplicate_groups, hamming_matrix, perceptual_hash


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


if __name__ == "__main__":
    unittest.main()
