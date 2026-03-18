from pathlib import Path

import pandas as pd

from privacy_utility_framework.utils.utils import dynamic_train_test_split

BASE_DIR = Path(__file__).parent


def _get_path(filename: str) -> str:
    return str(BASE_DIR.parent / filename)


def train_test_example():
    # Define the original dataset name
    orig = "insurance"

    # Specify the folder for saving train and test datasets
    folder = _get_path(f"examples/{orig}_datasets")

    # Load the original dataset from the specified path
    data = pd.read_csv(_get_path(f"datasets/original/{orig}.csv"), delimiter=",")
    # Split the data into train and test sets using dynamic_train_test_split function
    train, test = dynamic_train_test_split(data)

    print(f"~~~~~Train Dataset with {len(train)} rows~~~~~\n {train.head().to_string(index=False)}")
    print(f"~~~~~Test Dataset with {len(test)} rows~~~~~\n {test.head().to_string(index=False)}")
    # Save the train and test datasets to CSV files
    train.to_csv(f"{folder}/train/{orig}.csv", index=False)
    test.to_csv(f"{folder}/test/{orig}.csv", index=False)


train_test_example()
