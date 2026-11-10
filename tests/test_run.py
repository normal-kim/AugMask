"""run(): one method end to end on a toy table (train, sample, evaluate, write results)."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("sdmetrics")
import augmask.run as run_module  # noqa: E402

from toy import make_table  # noqa: E402

TINY = dict(n_layers_hidden=1, n_units_hidden=16, dim_emb=8, num_timesteps=5, batch_size=64, train_steps=4)


@pytest.mark.parametrize("fills", [1, 2])
def test_run_with_fills(tmp_path, monkeypatch, fills):
    monkeypatch.setattr(run_module, "load_dataset", lambda name: make_table("binclass"))
    metrics = run_module.run("toy", "mcar", 0.5, "tabddpm-augmask", out=tmp_path / "runs",
                             artifacts=tmp_path / "artifacts", device="cpu", fills=fills, **TINY)
    assert 0 <= metrics["shape"] <= 1 and metrics["checks"]["nan"] == 0 and metrics["fills"] == fills
    saved = json.loads((tmp_path / "runs/toy/mcar/p0.5/seed0/tabddpm-augmask.json").read_text())
    assert saved["shape"] == metrics["shape"]
    art = run_module.load_artifact(run_module.artifact_path(tmp_path / "artifacts", "toy", "mcar", 0.5, 0))
    assert run_module.artifact_fills(art) == fills
