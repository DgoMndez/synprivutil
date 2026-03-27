from pathlib import Path

from privacy_utility_framework.cyberdata import (
    DATASET_BY_KEY,
    dataset_destination,
    parse_assignment_flags,
    select_datasets,
)


def test_parse_assignment_flags_accepts_known_dataset_keys() -> None:
    overrides = parse_assignment_flags(
        [
            "swat=owner/swat-dataset",
            "unsw=https://example.com/UNSW_NB15_training-set.csv",
        ]
    )

    assert overrides == {
        "swat": "owner/swat-dataset",
        "unsw": "https://example.com/UNSW_NB15_training-set.csv",
    }


def test_select_datasets_defaults_to_full_manifest() -> None:
    selected = select_datasets(None)

    assert [dataset.key for dataset in selected] == ["rtn", "swat", "unsw"]


def test_dataset_destination_targets_cybersecurity_folder() -> None:
    root = Path("/tmp/synprivutil")
    destination = dataset_destination(root, DATASET_BY_KEY["swat"])

    assert destination == (root / "datasets" / "original" / "cybersecurity" / "swat-attack.csv")
