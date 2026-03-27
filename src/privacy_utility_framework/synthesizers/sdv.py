from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from .core import BaseModel


class GaussianCopulaModel(BaseModel):
    """
    A model for synthetic data generation using the Gaussian Copula method.
    Specifies GaussianCopulaSynthesizer as the synthesizer.
    """

    synthesizer_class = GaussianCopulaSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes GaussianCopulaModel with a GaussianCopulaSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(GaussianCopulaSynthesizer(metadata))


class CTGANModel(BaseModel):
    """
    A model for synthetic data generation using the CTGAN approach.
    Specifies CTGANSynthesizer as the synthesizer.
    """

    synthesizer_class = CTGANSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes CTGANModel with a CTGANSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(CTGANSynthesizer(metadata))


class CopulaGANModel(BaseModel):
    """
    A model for synthetic data generation using the Copula GAN approach.
    Specifies CopulaGANSynthesizer as the synthesizer.
    """

    synthesizer_class = CopulaGANSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes CopulaGANModel with a CopulaGANSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(CopulaGANSynthesizer(metadata))


class TVAEModel(BaseModel):
    """
    A model for synthetic data generation using the TVAE approach.
    Specifies TVAESynthesizer as the synthesizer.
    """

    synthesizer_class = TVAESynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes TVAEModel with a TVAESynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(TVAESynthesizer(metadata))
