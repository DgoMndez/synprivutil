import pandas as pd

from privacy_utility_framework.metrics import MetricFactory, MetricManager, MetricRegistry


def _build_dataframes():
    original = pd.DataFrame(
        {
            "age": [21, 35, 42, 50, 64],
            "income": [28000.0, 42000.0, 53000.0, 61000.0, 72000.0],
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": [22, 34, 43, 49, 63],
            "income": [30000.0, 40000.0, 52000.0, 60000.0, 71000.0],
        }
    )
    return original, synthetic


def test_metric_registry_factory_manager_flow_without_subclass_imports():
    original, synthetic = _build_dataframes()

    MetricRegistry.clear()
    MetricRegistry.discover()

    keys = MetricRegistry.list_registered()
    assert "basic_stats" in keys
    assert "utility.basic_stats" in keys

    single_result = MetricFactory.evaluate(
        "basic_stats",
        original=original,
        synthetic=synthetic,
    )
    assert isinstance(single_result, dict)
    assert set(single_result.keys()) == {"mean", "median", "var"}

    manager = MetricManager()
    manager.add_metric_by_name(
        "basic_stats",
        original=original,
        synthetic=synthetic,
    )
    manager_results = manager.evaluate_all()
    assert len(manager_results) == 1
    manager_value = next(iter(manager_results.values()))
    assert isinstance(manager_value, dict)
    assert set(manager_value.keys()) == {"mean", "median", "var"}

    batch_results = MetricFactory.evaluate_many(
        [
            {
                "id": "u-basic-stats",
                "name": "utility.basic_stats",
                "kwargs": {"original": original, "synthetic": synthetic},
            }
        ]
    )
    assert "u-basic-stats" in batch_results
    assert set(batch_results["u-basic-stats"].keys()) == {"mean", "median", "var"}
