import os
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


class Dataset:
    """
    A class to handle dataset splitting and verification for machine learning purposes.
    """

    @staticmethod
    def split_data(
            input_folder: str | Path,
            output_folder: str | Path,
            test_size: float = 0.2,
            random_state: int | float = None
    ):
        """
        Splits data from the input folder into training and testing datasets and copies them to the output folder.

        Parameters:
        input_folder (str | Path): The path to the folder containing the input data.
        output_folder (str | Path): The path to the folder where the split data will be saved.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2 (20%).
        random_state (int | float, optional): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output.

        Returns:
        None
        """
        train_folder = os.path.join(output_folder, 'train')
        test_folder = os.path.join(output_folder, 'test')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        class_files = {}

        for root, dirs, files in os.walk(input_folder):
            for class_dir in dirs:
                class_path = os.path.join(root, class_dir)
                files_list = os.listdir(class_path)
                class_files[class_dir] = [os.path.join(class_path, file_name) for file_name in files_list]

        min_images_per_class = 5000
        train_paths = []
        test_paths = []

        for class_dir, files in class_files.items():
            files = files[:min_images_per_class]
            train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

            # Move files to the corresponding folders in the output directory
            for src in train_files:
                dst = os.path.join(train_folder, class_dir, os.path.basename(src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
                train_paths.append(dst)

            for src in test_files:
                dst = os.path.join(test_folder, class_dir, os.path.basename(src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
                test_paths.append(dst)

    @classmethod
    def check_dataset(cls, path: Path | str) -> bool:
        """
        Checks if the dataset structure is valid by verifying that 'train' and 'test' folders exist and are not empty.

        Parameters:
        path (Path | str): The path to the folder containing the split dataset.

        Returns:
        bool: True if the dataset structure is valid, False otherwise.
        """
        if isinstance(path, str):
            path = Path(path)

        def is_not_empty(file_path: Path) -> bool:
            """
            Checks if a directory is not empty.

            Parameters:
            file_path (Path): The path to the directory.

            Returns:
            bool: True if the directory is not empty, False otherwise.
            """
            return len(list(file_path.glob('*'))) > 0

        expected_folders = ['train', 'test']
        if not all((path / folder).exists() for folder in expected_folders):
            raise FileNotFoundError("Expected folders 'train' and 'test' not found.")

        for folder in expected_folders:
            sub_folder_path = path / folder
            sub_folders = [sub_folder for sub_folder in sub_folder_path.iterdir() if sub_folder.is_dir()]

            if not all(is_not_empty(sub_folder) for sub_folder in sub_folders):
                return False

        return True


if __name__ == "__main__":
    input_folder = r"path_to_input_folder"
    output_folder = r"path_to_output_folder"
    test_size = 0.3  # 30% for testing
    random_state = 42  # Set seed for reproducibility (optional)

    Dataset.split_data(input_folder=input_folder,
                       output_folder=output_folder,
                       test_size=test_size,
                       random_state=random_state)

    is_valid = Dataset.check_dataset(output_folder)
    print(f"Dataset structure is valid: {is_valid}")
