**This is the repository for the thesis work "Design and Implementation of a Platform for Privacy Assessment of Synthetic Data Generation with Generative AI"  done at Communication Systems Group, University of Zurich, supervised by Mr. Weijie Niu, Dr. Alberto Huertas Celdran and Prof. Dr. Burkhard Stiller.**

# Synthetic Data Privacy and Utility Framework

This project provides a python library for generating synthetic datasets and evaluating their privacy and utility. It includes tools for creating synthetic data, performing various privacy and utility analyses, and visualizing results through plots. The framework is designed to help researchers and developers assess the balance between data utility and privacy in synthetic datasets.

## Prerequisites

The python version 3.10 was used to develop this framework.
For the following packages, these versions were used:

- Numpy version: 1.26.4
- Pandas version: 2.2.2
- SDV version: 1.15.0
- Scikit-learn version: 1.5.1
- Seaborn version: 0.12.2
- Matplotlib version: 3.9.2
- RDT version: 1.12.3
- Anonymeter version: 1.0.0
- Scipy version: 1.13.0
- Dython version: 0.7.8
- OT version: 0.9.4

## Installation

No other installation, besides cloning this repository is needed.

## Content

- datasets: in this folder, the original and synthetic datasets can be found. The following original datasets were used: [Diabetes](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset), [Cardio](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and [Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance).
- examples:
  - dataset_transform_normalization.py: Example of how to transform and normalize a dataset.
  - plots.py: Some plot functions that show how plots can be generated.
  - privacy_attacks.py: Usage-Example for each attacker-based privacy metric.
  - privacy_distance.py: Usage-Example for each distance-based privacy metric.
  - synthetic_data_generation.py: Example on how to generate synthetic data.
  - train_test.py: Example of creating train and test datasets.
  - utility.py: Usage-Example for each utility metric.
- plots: includes different kind of generated plots from the original and synthetic datasets.
- privacy_utility_framework: includes the framework code.
  - privacy_utility_framework:
    - dataset: includes the implementation of the Dataset object used across the code.
    - metrics: includes all privacy and utility metrics.
    - plots: includes the code for the available plots.
    - synthesizers: includes the implementation of all synthetic data generation models.
    - utils: utility functions, includes only dynamic_train_test_split at the moment.
- synthetic_models: includes the saved fitted models.

## Example Usage

Below is an example of how to generate synthetic data using the `GaussianMixtureModel`.

```python
# Load original dataset
original_data = pd.read_csv('../examples/insurance_datasets/train/insurance.csv')

# Create metadata for the dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(original_data)

# Initialize the Gaussian Mixture Model with a max of 10 components
gmm_model = GaussianMixtureModel(max_components=10)
# Fit the model on the original data
gmm_model.fit(original_data)

# Generate synthetic data
synthetic_data = gmm_model.sample(len(original_data))

# Save synthetic data to a CSV file
gmm_model.save_sample("gmm_sample.csv", len(original_data))

print("Synthetic data generated and saved to gmm_sample.csv.")
```

Here is an example of how to use the `PrivacyMetricManager` to evaluate privacy metrics between original and synthetic datasets.

```python
original_data = pd.read_csv(f"../datasets/original/diabetes.csv")
synthetic_data = pd.read_csv(f"../datasets/synthetic/diabetes_datasets/ctgan_sample.csv")

original_name = "Diabetes"
synthetic_name = "CTGAN"

p = PrivacyMetricManager()

metric_list = \
    [
        DCRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        NNDRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        AdversarialAccuracyCalculator(original_data, synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name)
    ]
p.add_metric(metric_list)
results = p.evaluate_all()
for key, value in results.items():
    print(f"{key}: {value}")
```

Here is another example of how to use the `UtilityMetricManager` to evaluate utility metrics between original and synthetic datasets. This example demonstrates the use of basic statistics and mutual information metrics.

```python
# Load original and synthetic datasets
original_data = pd.read_csv("../datasets/original/insurance.csv")
synthetic_data = pd.read_csv("../datasets/synthetic/insurance_datasets/ctgan_sample.csv")

# Specify dataset names for identification
original_name = "Insurance"
synthetic_name = "CTGAN"

# Initialize UtilityMetricManager
p = UtilityMetricManager()

# Define metrics to evaluate
metric_list = [
    BasicStatsCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
    MICalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
]

# Add metrics to manager and evaluate
p.add_metric(metric_list)
results = p.evaluate_all()

# Print results
for key, value in results.items():
    print(f"{key}: {value}")
```
