import pandas as pd

from privacy_utility_framework.dataset.dataset import DatasetManager


def dataset_example():
    # Sample original and synthetic data
    original_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["cat", "dog", "mouse"]})

    synthetic_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["cat", "dog", "mouse"]})
    # Initialize the DatasetManager
    manager = DatasetManager.from_dataframes(original_data, synthetic_data)

    # Set the transformer and scaler for the datasets
    manager.set_hypertransformer()

    # Transform and normalize the datasets
    manager.transform_datasets()

    # Access untouched original and synthetic data
    print("Original Data:\n", manager.original_dataset.data)
    print("Synthetic Data:\n", manager.synthetic_dataset.data)

    # Access transformed and normalized data
    print("Original Transformed Data:\n", manager.original_dataset.transformed_data)
    print("Synthetic Transformed Data:\n", manager.synthetic_dataset.transformed_data)


dataset_example()
