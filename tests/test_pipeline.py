"""Smoke tests de CI: validan la lógica del pipeline (split, arquitectura,
wrapper pyfunc) con datos sintéticos -- nada de esto pega contra Databricks
ni necesita el dataset real, así corre rápido en cualquier PR.

Uso: ./env_skin/bin/pytest tests/ -v
"""

import base64
import io
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from skin_classifier import (
    CLASS_NAMES,
    IMAGE_SIZE,
    build_model,
    get_transforms,
    train_valid_split,
)


def _fake_metadata(n_lesions=50, max_photos_per_lesion=3, seed=0):
    """Metadata sintética con la misma forma que HAM10000_metadata.csv:
    algunas lesiones tienen más de una foto (mismo lesion_id)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_lesions):
        lesion_id = f"HAM_{i:04d}"
        n_photos = rng.integers(1, max_photos_per_lesion + 1)
        dx = rng.choice(CLASS_NAMES)
        for j in range(n_photos):
            rows.append({"lesion_id": lesion_id, "image_id": f"ISIC_{i:04d}_{j}", "dx": dx})
    return pd.DataFrame(rows)


def _fake_image_b64(size=(32, 32), color=(120, 60, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestTrainValidSplit:
    def test_no_lesion_overlap(self):
        """Ninguna lesión (con potencialmente varias fotos) debe quedar
        repartida entre train y valid -- es la fuga de datos que el split
        por lesion_id existe para evitar."""
        df = _fake_metadata()
        train_df, valid_df = train_valid_split(df, seed=42, train_frac=0.8)
        train_lesions = set(train_df["lesion_id"])
        valid_lesions = set(valid_df["lesion_id"])
        assert train_lesions.isdisjoint(valid_lesions)

    def test_covers_all_rows(self):
        df = _fake_metadata()
        train_df, valid_df = train_valid_split(df, seed=42, train_frac=0.8)
        assert len(train_df) + len(valid_df) == len(df)

    def test_split_is_deterministic(self):
        df = _fake_metadata()
        t1, v1 = train_valid_split(df, seed=42)
        t2, v2 = train_valid_split(df, seed=42)
        assert set(t1["lesion_id"]) == set(t2["lesion_id"])
        assert set(v1["lesion_id"]) == set(v2["lesion_id"])

    def test_roughly_80_20_by_lesion(self):
        df = _fake_metadata(n_lesions=200)
        train_df, valid_df = train_valid_split(df, seed=42, train_frac=0.8)
        n_train_lesions = train_df["lesion_id"].nunique()
        n_valid_lesions = valid_df["lesion_id"].nunique()
        ratio = n_train_lesions / (n_train_lesions + n_valid_lesions)
        assert 0.75 <= ratio <= 0.85


class TestModelArchitecture:
    def test_output_shape(self):
        model = build_model(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, len(CLASS_NAMES))

    def test_transforms_output_shape(self):
        transform = get_transforms(train=False)
        img = Image.new("RGB", (64, 64))
        x = transform(img)
        assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


class TestPyfuncWrapper:
    def test_predict_shape_and_ranges(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from mlflow_model import SkinLesionPyfuncModel

        model = build_model(pretrained=False)
        weights_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), weights_path)

        class FakeContext:
            artifacts = {"weights": str(weights_path)}

        wrapper = SkinLesionPyfuncModel()
        wrapper.load_context(FakeContext())

        result = wrapper._predict_one(_fake_image_b64())

        assert set(result["probs"].keys()) == set(CLASS_NAMES)
        assert pytest.approx(sum(result["probs"].values()), abs=1e-4) == 1.0
        assert all(0.0 <= p <= 1.0 for p in result["probs"].values())

        grid = result["gradcam"]
        assert len(grid) == 7 and len(grid[0]) == 7  # layer4 de ResNet18 sobre 224x224
        flat = [v for row in grid for v in row]
        assert all(0.0 <= v <= 1.0 for v in flat)

    def test_predict_batch(self, tmp_path):
        from mlflow_model import SkinLesionPyfuncModel

        model = build_model(pretrained=False)
        weights_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), weights_path)

        class FakeContext:
            artifacts = {"weights": str(weights_path)}

        wrapper = SkinLesionPyfuncModel()
        wrapper.load_context(FakeContext())

        df = pd.DataFrame({"image_b64": [_fake_image_b64(), _fake_image_b64(color=(10, 200, 10))]})
        results = wrapper.predict(context=None, model_input=df)
        assert len(results) == 2
