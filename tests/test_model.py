"""Tests for the model and loss. Run from the repo root: python3 -m pytest tests

pretrained=False is used throughout so the tests never download weights.
"""

import unittest

import torch
from torch import nn

from src.model import build_model, class_weights, weighted_loss


class ModelTests(unittest.TestCase):
    def test_output_has_one_score_per_class(self):
        for arch in ("resnet18", "resnet50"):
            for num_classes in (3, 7):
                with self.subTest(arch=arch, num_classes=num_classes):
                    model = build_model(num_classes, arch=arch, pretrained=False).eval()
                    with torch.no_grad():
                        out = model(torch.randn(2, 3, 224, 224))
                    self.assertEqual(out.shape, (2, num_classes))

    def test_dropout_only_added_when_requested(self):
        self.assertIsInstance(build_model(3, arch="resnet18", pretrained=False).fc, nn.Linear)
        fc = build_model(3, arch="resnet18", pretrained=False, dropout=0.3).fc
        self.assertIsInstance(fc[0], nn.Dropout)
        self.assertEqual(fc[0].p, 0.3)

    def test_unknown_arch_rejected(self):
        with self.assertRaises(ValueError):
            build_model(3, arch="vgg16", pretrained=False)


class LossTests(unittest.TestCase):
    def test_balanced_counts_give_equal_weights(self):
        self.assertTrue(torch.allclose(class_weights([50, 50, 50]), torch.ones(3)))

    def test_rarer_class_gets_larger_weight(self):
        w = class_weights([10, 90])
        self.assertTrue(torch.allclose(w, torch.tensor([5.0, 100 / 180])))
        # Each class contributes equally in total: weight * count is constant.
        self.assertTrue(torch.allclose(w * torch.tensor([10.0, 90.0]), torch.tensor([50.0, 50.0])))

    def test_empty_class_rejected(self):
        with self.assertRaises(ValueError):
            class_weights([10, 0, 5])

    def test_loss_uses_weights(self):
        loss_fn = weighted_loss([10, 90])
        self.assertTrue(torch.allclose(loss_fn.weight, class_weights([10, 90])))
        loss = loss_fn(torch.randn(4, 2), torch.tensor([0, 1, 1, 0]))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
