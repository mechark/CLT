import kagglehub

class ProbingDatasetLoader:
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

    def load_dataset(self) -> list:
        """
        Loads the dataset from the specified filepath.
        
        Returns:
            data: The loaded dataset as a list of examples.
        """
        path = self.dataset_folder + "/data_set_4.csv"
        with open(path, 'r') as file:
            data = file.readlines()
        return data