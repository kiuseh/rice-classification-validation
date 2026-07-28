import json
from pathlib import Path

import pytest

from rice_model import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    run_experiment,
    split_dataset,
)

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "Rice_Cammeo_Osmancik.xlsx"


def test_official_dataset_shape_schema_and_classes() -> None:
    data = load_dataset(DATASET_PATH)

    assert data.shape == (3810, 8)
    assert list(data.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert data[TARGET_COLUMN].value_counts().to_dict() == {
        "Osmancik": 2180,
        "Cammeo": 1630,
    }
    assert data.isna().sum().sum() == 0
    assert data.duplicated().sum() == 0


def test_split_is_deterministic_and_stratified() -> None:
    data = load_dataset(DATASET_PATH)
    first_split = split_dataset(data)
    second_split = split_dataset(data)

    for first, second in zip(first_split, second_split, strict=True):
        assert first.index.tolist() == second.index.tolist()

    x_train, x_test, y_train, y_test = first_split
    assert len(x_train) == 3048
    assert len(x_test) == 762
    assert set(x_train.index).isdisjoint(x_test.index)

    full_ratio = data[TARGET_COLUMN].value_counts(normalize=True)
    train_ratio = y_train.value_counts(normalize=True)
    test_ratio = y_test.value_counts(normalize=True)
    for class_name in CLASS_NAMES:
        assert train_ratio[class_name] == pytest.approx(full_ratio[class_name], abs=0.001)
        assert test_ratio[class_name] == pytest.approx(full_ratio[class_name], abs=0.001)


def test_experiment_writes_consistent_artifacts(tmp_path: Path) -> None:
    results, model = run_experiment(
        DATASET_PATH,
        output_dir=tmp_path,
        cv_splits=3,
        n_estimators=60,
    )

    assert results["holdout"]["accuracy"] > 0.88
    assert results["holdout"]["macro_f1"] > 0.88
    assert sum(map(sum, results["holdout"]["confusion_matrix"])) == 762
    assert model.classes_.tolist() == CLASS_NAMES

    metrics_path = tmp_path / "metrics.json"
    figure_path = tmp_path / "evaluation-summary.png"
    assert metrics_path.is_file()
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 10_000

    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted["holdout"] == results["holdout"]
