# Datasets

This directory contains datasets used to test or simply experiment with the synprivutil library. The structure of the directory is as follows:

* `original/`: This subdirectory contains the original datasets, which are used as the basis for testing and experimentation. These datasets are typically in their raw form, without any modifications or preprocessing. Any additional subdirectory within `original/` may contain datasets from a specific topic like `cybersecurity/`.
* `synthetic/`: This subdirectory contains synthetic datasets, which are generated using the synprivutil library. There are different folders for each original dataset, and within those folders, there are different synthetic datasets named after its synthetic data generation method.

The cybersecurity CSVs in `original/cybersecurity/` are intended to be downloaded
locally with `privacy-framework-install-cyberdata` and are gitignored by default.
For `UNSW_NB15_training-set.csv`, the installer references the official UNSW dataset
page but uses a public mirror for the file download because the landing page does not
publish a stable direct CSV URL.

## References

1. [RTN_traffic_dataset.csv](original/cybersecurity/RTN_traffic_dataset.csv): Chaudhari, R., & Deshpande, M. M. (2026). *Real-Time Network Traffic Dataset for IDS* [Data set]. Kaggle. <https://doi.org/10.34740/KAGGLE/DSV/15092498>
2. [swat-attack.csv](original/cybersecurity/swat-attack.csv): Secure Water Treatment (SWaT) dataset, using the Kaggle-hosted `attack` partition. Source: <https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system>
3. [UNSW_NB15_training-set.csv](original/cybersecurity/UNSW_NB15_training-set.csv): The UNSW-NB15 dataset training partition. Source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Citation: Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).* MilCIS 2015, IEEE.
