import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from privacy_utility_framework.dataset.dataset import Dataset


class BaseModel:
    """
    Base class for models that use a synthesizer to generate synthetic data.

    Attributes:
        synthesizer_class (class): Placeholder for the synthesizer class, \
            to be specified in subclasses.
    """

    synthesizer_class = None

    def __init__(self, synthesizer):
        """
        Initializes the model with a specific synthesizer instance.

        Args:
            synthesizer: An instance of a synthesizer class used for synthetic data generation.
        """
        self.synthesizer = synthesizer

    def fit(self, data: pd.DataFrame) -> None:
        """
        Trains the synthesizer on the provided dataset.

        Args:
            data (pd.DataFrame): The data to be used for fitting the synthesizer.
        """
        self.synthesizer.fit(data)

    def sample(self, num_samples: int = 200) -> pd.DataFrame:
        """
        Generates synthetic samples using the trained synthesizer.

        Args:
            num_samples (int): The number of synthetic samples to generate (default is 200).

        Returns:
            pd.DataFrame: The generated synthetic samples.
        """
        return self.synthesizer.sample(num_samples)

    def save_sample(self, filename: str, num_samples: int = 200) -> None:
        """
        Generates and saves synthetic samples to a CSV file.

        Args:
            filename (str): The name of the file to save the synthetic data.
            num_samples (int): The number of synthetic samples to generate (default is 200).
        """
        synthetic_data = self.sample(num_samples)
        synthetic_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_model(self, filename: str) -> None:
        """
        Saves the current synthesizer model to a file, if supported by the synthesizer.

        Args:
            filename (str): The name of the file to save the model.

        Raises:
            AttributeError: If the synthesizer does not support the .save method.
        """
        if hasattr(self.synthesizer, "save"):
            self.synthesizer.save(filename)
            print(f"Model saved to {filename}")
        else:
            raise AttributeError("The current synthesizer does not support the 'save' method.")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Loads a saved synthesizer model from a specified file and returns a model instance,
        if supported by the synthesizer.

        Args:
            filepath (str): The path to the file containing the saved synthesizer model.

        Returns:
            An instance of the model with the synthesizer loaded from the specified file.

        Raises:
            AttributeError: If the synthesizer class does not support the .load method.
        """
        if hasattr(cls.synthesizer_class, "load"):
            synthesizer = cls.synthesizer_class.load(filepath)
            instance = cls.__new__(cls)
            instance.synthesizer = synthesizer
            return instance
        else:
            raise AttributeError(
                f"The synthesizer class '{cls.synthesizer_class.__name__}' \
                    does not support the 'load' method."
            )


class GaussianMixtureModel(BaseModel):
    """
    Model for generating synthetic data using a Gaussian Mixture Model (GMM).
    Performs data transformation and fitting with optional selection of optimal components.
    """

    def __init__(self, max_components: int = 10):
        """
        Initializes GaussianMixtureModel with a maximum number of components to test for GMM.

        Args:
            max_components (int): The maximum number of components to consider.
        """
        super().__init__(None)
        self.transformed_data = None
        self.transformer = None
        self.max_components = max_components
        self.model = None

    def fit(self, data: pd.DataFrame, random_state: int = 42) -> None:
        """
        Transforms data, selects optimal components, and fits the GMM.

        Args:
            data (pd.DataFrame): Input data for model fitting.
            random_state (int): Seed for reproducibility.
        """
        dataset = Dataset(data)
        dataset.fit_tabletransformer()
        dataset.transform()
        self.transformed_data = dataset.transformed_data
        self.transformer = dataset.tabletransformer
        optimal_n_components = self._select_n_components(self.transformed_data, random_state)
        self.model = GaussianMixture(n_components=optimal_n_components, random_state=random_state)
        self.model.fit(self.transformed_data)

    def sample(self, num_samples: int = 200) -> pd.DataFrame:
        """
        Generates synthetic samples by sampling from the fitted GMM.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Synthetic samples in the original data format.
        """
        if self.model is not None:
            samples, _ = self.model.sample(num_samples)
            samples_pd = pd.DataFrame(samples, columns=self.transformed_data.columns)
            inverse_samples = self.transformer.reverse_transform(samples_pd)
            return inverse_samples
        else:
            raise RuntimeError("Data has not been fitted yet.")

    def save_model(self, filename: str) -> None:
        raise NotImplementedError("GaussianMixtureModel does not support model saving.")

    @classmethod
    def load_model(cls, filepath: str):
        raise NotImplementedError("GaussianMixtureModel does not support model loading.")

    def _select_n_components(self, data: pd.DataFrame, random_state: int) -> int:
        """
        Selects the optimal number of GMM components using the Bayesian Information Criterion (BIC).

        Args:
            data (pd.DataFrame): Dataset for fitting GMM models.
            random_state (int): Seed for reproducibility.

        Returns:
            int: Optimal number of components based on BIC.
        """
        bics = []
        n_components_range = range(1, self.max_components + 1)

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            gmm.fit(data)
            bics.append(gmm.bic(data))

        optimal_n_components = n_components_range[np.argmin(bics)]
        return optimal_n_components


class RandomModel(BaseModel):
    """
    Model that generates synthetic data by randomly sampling from a given dataset.
    This model does not require complex fitting.
    """

    def __init__(self):
        """
        Initializes RandomModel with no synthesizer and sets default attributes.
        """
        super().__init__(None)
        self.data = None
        self.trained = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Sets the provided dataset as the source for random sampling.

        Args:
            data (pd.DataFrame): The dataset to be used for random sampling.
        """
        self.trained = True
        self.data = data

    def sample(self, num_samples: int = None, random_state: int = None) -> pd.DataFrame:
        """
        Randomly samples data points from the dataset.

        Args:
            num_samples (int): Number of samples to generate.
            random_state (int): Seed for reproducibility.

        Returns:
            pd.DataFrame: Containing randomly sampled data.

        Raises:
            RuntimeError if the model has not been fitted with a dataset.
        """
        if self.trained:
            if num_samples is None:
                return self.data
            return pd.DataFrame(
                self.data.sample(num_samples, random_state=random_state, replace=False)
            )
        else:
            raise RuntimeError("No dataset provided to generator")

    def save_model(self, filename: str) -> None:
        raise NotImplementedError("RandomModel does not support model saving.")

    @classmethod
    def load_model(cls, filepath: str):
        raise NotImplementedError("RandomModel does not support model loading.")
