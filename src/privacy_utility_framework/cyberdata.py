"""
Command-line utility to download the original cybersecurity datasets into \
    `datasets/original/cybersecurity/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_DIR = Path("datasets") / "original" / "cybersecurity"
DEFAULT_USER_AGENT = "synprivutil-cyberdata-installer/1.0"
KAGGLE_AUTH_HELP = (
    "Kaggle downloads require ~/.kaggle/kaggle.json or the "
    "KAGGLE_USERNAME/KAGGLE_KEY environment variables."
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    description: str
    source_url: str
    download_kind: str
    kaggle_ref: str | None = None
    kaggle_search: str | None = None
    download_url: str | None = None
    notes: str = ""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="rtn",
        filename="RTN_traffic_dataset.csv",
        description="Real-Time Network Traffic Dataset for IDS",
        source_url="https://doi.org/10.34740/KAGGLE/DSV/15092498",
        download_kind="kaggle",
        kaggle_search="Real-Time Network Traffic Dataset for IDS",
        notes=(
            "The DOI points to Kaggle but does not expose a stable API slug, so the "
            "installer resolves this dataset through the Kaggle API search endpoint."
        ),
    ),
    DatasetSpec(
        key="swat",
        filename="swat-attack.csv",
        description="Secure Water Treatment (SWaT) attack partition",
        source_url=(
            "https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system"
        ),
        download_kind="kaggle",
        kaggle_ref="vishala28/swat-dataset-secure-water-treatment-system",
    ),
    DatasetSpec(
        key="unsw",
        filename="UNSW_NB15_training-set.csv",
        description="UNSW-NB15 training partition",
        source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        download_kind="direct_url",
        download_url=(
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/"
            "UNSW_NB15_training-set.csv"
        ),
        notes=(
            "The official UNSW page is used as the authoritative source reference. "
            "The automated installer fetches the training split from a public mirror "
            "because the UNSW landing page does not publish a stable direct CSV URL."
        ),
    ),
)

DATASET_BY_KEY = {dataset.key: dataset for dataset in DATASETS}


def find_repo_root(explicit_root: str | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    candidates = [Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for path in (current, *current.parents):
            if (path / "pyproject.toml").is_file() and (path / "datasets").is_dir():
                return path

    raise FileNotFoundError(
        "Could not locate the synprivutil repository root. Pass --root explicitly."
    )


def parse_assignment_flags(values: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid assignment '{value}'. Use the form dataset=value.")
        key, assigned_value = value.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key not in DATASET_BY_KEY:
            valid = ", ".join(sorted(DATASET_BY_KEY))
            raise ValueError(f"Unknown dataset '{normalized_key}'. Valid dataset keys: {valid}.")
        if not assigned_value.strip():
            raise ValueError(f"Missing value for dataset '{normalized_key}'.")
        assignments[normalized_key] = assigned_value.strip()
    return assignments


def select_datasets(selected: list[str] | None) -> list[DatasetSpec]:
    if not selected or "all" in selected:
        return list(DATASETS)

    resolved: list[DatasetSpec] = []
    seen: set[str] = set()
    for key in selected:
        if key in seen:
            continue
        resolved.append(DATASET_BY_KEY[key])
        seen.add(key)
    return resolved


def dataset_destination(root: Path, dataset: DatasetSpec) -> Path:
    return root / DEFAULT_DATASET_DIR / dataset.filename


def load_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Kaggle support is not installed. Run 'pip install -e .[cyberdata]'."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        raise RuntimeError(KAGGLE_AUTH_HELP) from exc
    return api


def resolve_kaggle_ref(api, dataset: DatasetSpec, overrides: dict[str, str]) -> str:
    if dataset.key in overrides:
        return overrides[dataset.key]
    if dataset.kaggle_ref is not None:
        return dataset.kaggle_ref
    if dataset.kaggle_search is None:
        raise RuntimeError(f"No Kaggle reference configured for dataset '{dataset.key}'.")

    results = list(api.dataset_list(search=dataset.kaggle_search))
    if not results:
        raise RuntimeError(f"No Kaggle dataset matched search query '{dataset.kaggle_search}'.")

    exact_matches = [
        item
        for item in results
        if getattr(item, "title", "").strip().casefold() == dataset.description.strip().casefold()
    ]
    selected = exact_matches[0] if exact_matches else results[0]
    ref = getattr(selected, "ref", None)
    if not ref:
        raise RuntimeError(
            f"Kaggle search for '{dataset.description}' returned an entry without a ref."
        )
    return ref


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def describe_kaggle_source(
    dataset: DatasetSpec,
    kaggle_overrides: dict[str, str],
) -> str:
    if dataset.key in kaggle_overrides:
        return kaggle_overrides[dataset.key]
    if dataset.kaggle_ref is not None:
        return dataset.kaggle_ref
    if dataset.kaggle_search is not None:
        return f"Kaggle search: {dataset.kaggle_search}"
    return "Kaggle dataset"


def download_kaggle_file(
    api,
    dataset: DatasetSpec,
    destination: Path,
    kaggle_overrides: dict[str, str],
    force: bool,
    dry_run: bool,
) -> str:
    if dry_run:
        kaggle_source = describe_kaggle_source(dataset, kaggle_overrides)
        return f"[dry-run] {dataset.key}: would download {dataset.filename} from {kaggle_source}"

    kaggle_ref = resolve_kaggle_ref(api, dataset, kaggle_overrides)

    with tempfile.TemporaryDirectory(prefix=f"synprivutil-{dataset.key}-") as tmpdir:
        api.dataset_download_files(
            kaggle_ref,
            path=tmpdir,
            unzip=True,
            quiet=False,
            force=force,
        )
        candidates = sorted(Path(tmpdir).rglob(dataset.filename))
        if not candidates:
            raise FileNotFoundError(
                f"Downloaded Kaggle archive for '{dataset.key}' does not contain "
                f"{dataset.filename}."
            )
        copy_file(candidates[0], destination)

    return f"[ok] {dataset.key}: downloaded {dataset.filename} from {kaggle_ref}"


def download_direct_file(
    dataset: DatasetSpec,
    destination: Path,
    direct_overrides: dict[str, str],
    dry_run: bool,
) -> str:
    download_url = direct_overrides.get(dataset.key, dataset.download_url)
    if download_url is None:
        raise RuntimeError(f"No direct download URL configured for dataset '{dataset.key}'.")
    if dry_run:
        return f"[dry-run] {dataset.key}: would download {dataset.filename} from {download_url}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    return f"[ok] {dataset.key}: downloaded {dataset.filename} from {download_url}"


def install_datasets(
    root: Path,
    datasets: list[DatasetSpec],
    kaggle_overrides: dict[str, str],
    direct_overrides: dict[str, str],
    force: bool,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    output: list[str] = []
    errors: list[str] = []
    kaggle_api = None

    for dataset in datasets:
        destination = dataset_destination(root, dataset)
        if destination.exists() and not force:
            output.append(f"[skip] {dataset.key}: {destination} already exists")
            continue

        try:
            if dataset.download_kind == "kaggle":
                if not dry_run and kaggle_api is None:
                    kaggle_api = load_kaggle_api()
                message = download_kaggle_file(
                    api=kaggle_api,
                    dataset=dataset,
                    destination=destination,
                    kaggle_overrides=kaggle_overrides,
                    force=force,
                    dry_run=dry_run,
                )
            elif dataset.download_kind == "direct_url":
                message = download_direct_file(
                    dataset=dataset,
                    destination=destination,
                    direct_overrides=direct_overrides,
                    dry_run=dry_run,
                )
            else:  # pragma: no cover - guarded by static manifest
                raise RuntimeError(f"Unsupported download kind '{dataset.download_kind}'.")

            output.append(message)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[error] {dataset.key}: {exc}")

    return output, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the optional cybersecurity datasets into datasets/original/cybersecurity/."
        )
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root. Defaults to auto-detecting the synprivutil checkout.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["all", *sorted(DATASET_BY_KEY)],
        help="Dataset key to install. Repeat to install a subset. Defaults to all.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        action="append",
        default=[],
        metavar="DATASET=OWNER/SLUG",
        help=(
            "Override the Kaggle dataset ref for a specific dataset. Useful when the "
            "citation only exposes a DOI and not a stable Kaggle API slug."
        ),
    )
    parser.add_argument(
        "--direct-url",
        action="append",
        default=[],
        metavar="DATASET=URL",
        help="Override the direct download URL for a specific dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if the target file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned downloads without fetching any files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        kaggle_overrides = parse_assignment_flags(args.kaggle_dataset)
        direct_overrides = parse_assignment_flags(args.direct_url)
        root = find_repo_root(args.root)
        datasets = select_datasets(args.dataset)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))

    print(f"Installing cybersecurity datasets into {root / DEFAULT_DATASET_DIR}")
    for dataset in datasets:
        print(f"- {dataset.key}: {dataset.description}")
        print(f"  source: {dataset.source_url}")
        if dataset.notes:
            print(f"  note: {dataset.notes}")

    output, errors = install_datasets(
        root=root,
        datasets=datasets,
        kaggle_overrides=kaggle_overrides,
        direct_overrides=direct_overrides,
        force=args.force,
        dry_run=args.dry_run,
    )

    if output:
        print()
        for line in output:
            print(line)

    if errors:
        print(file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
