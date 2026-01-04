import kagglehub
from abc import ABC, abstractmethod
import pandas as pd
from datasets import load_dataset, load_from_disk

class DatasetLoader(ABC):
    
    @abstractmethod
    def download_dataset(self, dataset_name) -> str:
        """Downloads and loads the dataset from the specified filepath.
        
        Returns:
        path: Path to the downloaded dataset.
        """
        pass

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Loads the dataset from the specified filepath.
        
        Returns:
        data: The loaded dataset as a list of examples.
        """
        pass


class KaggleProbingDatasetLoader(DatasetLoader):
    """Class to download and load the probing dataset from Kaggle."""
    def download_dataset(self, dataset_name) -> str:
        """
        Downloads and loads the dataset from the specified filepath.
        
        Returns:
            path: Path to the downloaded dataset.
        """
        path = kagglehub.dataset_download(dataset_name)
        self.dataset_folder = path
        return path

    def load_dataset(self, dataset_name) -> pd.DataFrame:
        """
        Loads the dataset from the specified filepath.
        
        Returns:
            data: The loaded dataset as a list of examples.
        """
        path = self.dataset_folder + '\\' + dataset_name
        data = pd.read_csv(path)
        return data
    
class HuggingFaceDatasetLoader(DatasetLoader):
    """Class to download and load the SQuAD dataset from Hugging Face."""
    def __init__(self):
        self.dataset_folder = None

    def download_dataset(self, dataset_name, split="train") -> str:
        ds = load_dataset(dataset_name, split=split)
        ds.save_to_disk(f".datasets/{dataset_name}_{split}")
        self.dataset_folder = f".datasets/{dataset_name}_{split}"

    def load_dataset(self):
        return load_dataset(self.dataset_folder)
    
    def load_dataset(self, dataset_folder):
        return load_from_disk(dataset_folder)