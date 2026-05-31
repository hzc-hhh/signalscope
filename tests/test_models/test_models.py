"""
Tests for SignalScope model modules.
"""

import pytest
import torch

from signalscope.models import ModelZoo, ResNet1D, TransformerTS


class TestResNet1D:
    def test_creation(self):
        model = ResNet1D(in_channels=1, num_classes=1)
        assert model.name == "ResNet1D"
        assert model.num_classes == 1

    def test_forward(self):
        model = ResNet1D(in_channels=1, num_classes=1)
        x = torch.randn(4, 1, 500)
        out = model(x)
        assert out.shape == (4,)

    def test_forward_batch(self):
        model = ResNet1D(in_channels=1, num_classes=2)
        x = torch.randn(8, 1, 200)
        out = model(x)
        assert out.shape == (8, 2)


class TestTransformerTS:
    def test_creation(self):
        model = TransformerTS(d_model=64, nhead=4, num_layers=2)
        assert model.name == "TransformerTS"

    def test_forward(self):
        model = TransformerTS(d_model=64, nhead=4, num_layers=2, num_classes=1)
        x = torch.randn(2, 500)
        out = model(x)
        assert out.shape == (2,)

    def test_cls_token(self):
        model = TransformerTS(d_model=128, num_classes=1)
        x = torch.randn(1, 300)
        out = model(x)
        assert out.shape == (1,)


class TestModelZoo:
    def test_list_models(self):
        zoo = ModelZoo()
        models = zoo.list()
        assert "resnet1d" in models
        assert "transformer_ts" in models

    def test_get_model(self):
        zoo = ModelZoo()
        model = zoo.get("resnet1d", in_channels=1, num_classes=1)
        assert isinstance(model, ResNet1D)

    def test_get_unknown_model(self):
        zoo = ModelZoo()
        with pytest.raises(KeyError):
            zoo.get("nonexistent_model")
