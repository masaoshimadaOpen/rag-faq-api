"""The eval CLI runs end-to-end on the shipped dataset, offline."""
from __future__ import annotations

from pathlib import Path

from app.evaluate import load_dataset, main, _DEFAULT_DATASET


def test_default_dataset_exists_and_loads():
    documents, dataset = load_dataset(_DEFAULT_DATASET)
    assert len(documents) >= 3
    assert len(dataset) >= 3
    assert all(ex.relevant_ids for ex in dataset)


def test_cli_main_runs_and_reports(capsys):
    rc = main([])  # no args → default dataset
    assert rc == 0
    out = capsys.readouterr().out
    assert "Retrieval evaluation" in out
    assert "Chunking strategy sweep" in out
    assert "best:" in out


def test_cli_missing_dataset_returns_1(capsys):
    rc = main([str(Path("does_not_exist_12345.json"))])
    assert rc == 1
