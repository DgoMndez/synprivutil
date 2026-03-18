"""
Module: examples/privacy_distance.py
Description: Example script for privacy distance metrics using PrivacyUtilityFramework.

Creation Date: 30/10/2024
"""

import time
import warnings
from pathlib import Path

import pandas as pd

from privacy_utility_framework.dataset.dataset import Dataset
from privacy_utility_framework.metrics.privacy_metrics.distance.adversarial_accuracy_class import (
    AdversarialAccuracyCalculator,
    AdversarialAccuracyCalculator_NN,
)
from privacy_utility_framework.metrics.privacy_metrics.distance.dcr_class import (
    DCRCalculator,
)
from privacy_utility_framework.metrics.privacy_metrics.distance.disco import (
    DisclosureCalculator,
)
from privacy_utility_framework.metrics.privacy_metrics.distance.nndr_class import (
    NNDRCalculator,
)
from privacy_utility_framework.metrics.privacy_metrics.privacy_metric_manager import (
    PrivacyMetricManager,
)

RANDOM_STATE = 7428
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent / "datasets"

warnings.filterwarnings("ignore", category=FutureWarning)


def _dataset_path(*parts: str) -> Path:
    return DATASETS_DIR.joinpath(*parts)


def dcr_example():
    print("~~~~~~~~~DCR EXAMPLE~~~~~~~~~~")
    original_data = pd.read_csv(_dataset_path("original", "diabetes.csv"))
    synthetic_data = pd.read_csv(
        _dataset_path("synthetic", "diabetes_datasets", "ctgan_sample.csv")
    )

    test_dcr_calculator = DCRCalculator(
        original_data, synthetic_data, weights=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    test_dcr = test_dcr_calculator.evaluate()
    print(f"DCR (diabetes, ctgan): {test_dcr}")


def nndr_example():
    print("~~~~~~~~~NNDR EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets = ["diabetes", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(_dataset_path("original", f"{orig}.csv"))
            synthetic_data = pd.read_csv(
                _dataset_path("synthetic", f"{orig}_datasets", f"{syn}_sample.csv")
            )
            test_nndr_calculator = NNDRCalculator(original_data, synthetic_data)
            test_nndr = test_nndr_calculator.evaluate()
            print(f"NNDR {orig, syn}: {test_nndr}")


def nnaa_example():
    print("~~~~~~~~~NNAA EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets = ["diabetes", "insurance", "cardio"]

    for i, orig in enumerate(original_datasets):
        for j, syn in enumerate(synthetic_datasets):
            original_data = pd.read_csv(_dataset_path("original", f"{orig}.csv"))
            synthetic_data = pd.read_csv(
                _dataset_path("synthetic", f"{orig}_datasets", f"{syn}_sample.csv")
            )

            if i == 2 and j > 0:
                continue

            print(f"~~~~~~Adversarial Accuracy CDIST~~~~~~ {orig, syn}")

            try:
                t0 = time.time()
                calculator_cdist = AdversarialAccuracyCalculator(original_data, synthetic_data)
                nnaa1 = calculator_cdist.evaluate()
                t1 = time.time()
                print(nnaa1)
                print(f"Time taken for CDIST NNAA: {t1 - t0:.2f} seconds")
            except Exception as e:
                print(f"Error calculating NNAA with CDIST for {orig, syn}: {e}")

            print(f"~~~~~~Adversarial Accuracy NN~~~~~~ {orig, syn}")
            calculator_nn = AdversarialAccuracyCalculator_NN(original_data, synthetic_data)
            nnaa2 = calculator_nn.evaluate()
            print(nnaa2)


def disco_example():
    print("~~~~~~~~~DiSCO/REPU EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    # original_datasets = ["diabetes"]
    # diabetes_keys = ["Age", "BMI", "DiabetesPedigreeFunction", "Glucose", "BloodPressure"]
    # diabetes_target = "Outcome"

    # Other examples, adjust parameters below

    # original_datasets =["cardio"]
    # cardio_keys = ['age', 'gender', 'height', 'weight', 'cholesterol', 'gluc']
    # cardio_target = 'cardio'
    #
    #
    original_datasets = ["insurance"]
    insurance_keys = ["age", "bmi", "children"]
    insurance_target = "charges"

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(_dataset_path("original", f"{orig}.csv"))
            synthetic_data = pd.read_csv(
                _dataset_path("synthetic", f"{orig}_datasets", f"{syn}_sample.csv")
            )
            print(f"~~~Pair: {orig, syn}~~~\n")
            calc = DisclosureCalculator(
                original_data, synthetic_data, keys=insurance_keys, target=insurance_target
            )
            repU, DiSCO = calc.evaluate()
            print(f"repU: {repU}, DiSCO: {DiSCO}")


def privacy_metric_manager_example():
    original_data = pd.read_csv(_dataset_path("original", "diabetes.csv"))
    synthetic_data = pd.read_csv(
        _dataset_path("synthetic", "diabetes_datasets", "ctgan_sample.csv")
    )
    original_name = "Diabetes"
    synthetic_name = "CTGAN"
    p = PrivacyMetricManager()
    metric_list = [
        DCRCalculator(
            original_data,
            synthetic_data,
            original_name=original_name,
            synthetic_name=synthetic_name,
        ),
        NNDRCalculator(
            original_data,
            synthetic_data,
            original_name=original_name,
            synthetic_name=synthetic_name,
        ),
        AdversarialAccuracyCalculator(
            original_data,
            synthetic_data,
            original_name=original_name,
            synthetic_name=synthetic_name,
        ),
    ]
    p.add_metric(metric_list)
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")


def privacy_metric_manager_quantile_example():
    print("~~~~~~~~~PRIVACY METRICS (QUANTILE DISTANCE)~~~~~~~~~~")
    datasets = ["diabetes", "insurance"]
    synthetizers = ["copulagan", "ctgan"]

    for orig in datasets:
        for syn in synthetizers:
            original_data = pd.read_csv(_dataset_path("original", f"{orig}.csv"))
            synthetic_data = pd.read_csv(
                _dataset_path("synthetic", f"{orig}_datasets", f"{syn}_sample.csv")
            )
            original_name = orig.capitalize()
            synthetic_name = syn.upper()

            print(f"~~~Pair: {orig, syn}~~~\n")

            original_dataset = Dataset(original_data, name=original_name)
            original_dataset.fit_transform()

            quantile_metric_args = {
                "original_data": original_dataset.transformed_data,
                "base_metric": "euclidean",
                "output_distribution": "uniform",
                "n_quantiles": min(100, len(original_dataset.transformed_data)),
            }

            p = PrivacyMetricManager()
            metric_list = [
                DCRCalculator(
                    original_data,
                    synthetic_data,
                    original_name=original_name,
                    synthetic_name=synthetic_name,
                    distance_strategy="quantile",
                    **quantile_metric_args,
                ),
                NNDRCalculator(
                    original_data,
                    synthetic_data,
                    original_name=original_name,
                    synthetic_name=synthetic_name,
                    distance_strategy="quantile",
                    **quantile_metric_args,
                ),
                AdversarialAccuracyCalculator(
                    original_data,
                    synthetic_data,
                    original_name=original_name,
                    synthetic_name=synthetic_name,
                    distance_strategy="quantile",
                    **quantile_metric_args,
                ),
                AdversarialAccuracyCalculator_NN(
                    original_data,
                    synthetic_data,
                    original_name=original_name,
                    synthetic_name=synthetic_name,
                    distance_strategy="quantile",
                    **quantile_metric_args,
                ),
            ]
            p.add_metric(metric_list)
            results = p.evaluate_all()
            for key, value in results.items():
                print(f"{key}: {value}")


def privacy_metric_manager_ecdf_example():
    print("~~~~~~~~~PRIVACY METRICS (ECDF DISTANCE)~~~~~~~~~~")
    datasets = ["diabetes", "insurance", "cardio"]
    synthetizers = ["copulagan", "ctgan", "gmm", "tvae", "random"]

    for orig in datasets:
        for syn in synthetizers:
            original_data = pd.read_csv(_dataset_path("original", f"{orig}.csv"))
            synthetic_data = pd.read_csv(
                _dataset_path("synthetic", f"{orig}_datasets", f"{syn}_sample.csv")
            )
            original_name = orig.capitalize()
            synthetic_name = syn.upper()

            original_dataset = Dataset(original_data, name=original_name)
            original_dataset.fit_transform()

            metric_args = {
                "original_data": original_dataset.transformed_data,
                "base_metric": "euclidean",
            }

            print(f"~~~Pair: {orig, syn}~~~\n")

            p = PrivacyMetricManager()
            metric_list = [
                AdversarialAccuracyCalculator(
                    original_data,
                    synthetic_data,
                    original_name=original_name,
                    synthetic_name=synthetic_name,
                    distance_strategy="ecdf",
                    nn_samples=1000,
                    nn_random_state=RANDOM_STATE,
                    **metric_args,
                ),
            ]
            p.add_metric(metric_list)
            results = p.evaluate_all()
            for key, value in results.items():
                print(f"{key}: {value}")


dcr_example()
nndr_example()
nnaa_example()
privacy_metric_manager_example()
privacy_metric_manager_quantile_example()
privacy_metric_manager_ecdf_example()
