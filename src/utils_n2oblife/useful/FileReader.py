import os
import pickle
import shutil
import numpy as np

def copyfile(src, dst):
    shutil.copyfile(src, dst)

def save_data(data, filename:str, format='pickle'):
    """
    Save data to a file in the specified format.

    Args:
        data: The data to save.
        filename (str): The filename to save the data to.
        format (str): The format to save the data in. Default is 'pickle'.
    """
    if format == 'pickle':
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to {filename}")
    elif format == 'npy':
        np.save(filename, data)
        print(f"Data saved to {filename}")
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_data(filename, format='pickle'):
    """
    Load data from a file in the specified format.

    Args:
        filename (str): The filename to load the data from.
        format (str): The format to load the data in. Default is 'pickle'.

    Returns:
        The loaded data.
    """
    if format == 'pickle':
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print(f"Data loaded from {filename}")
        return data
    elif format == 'npy':
        data = np.load(filename)
        print(f"Data loaded from {filename}")
        return data
    else:
        raise ValueError(f"Unsupported format: {format}")

def check_files_exist(folder_path: str, filenames: list[str]) -> dict[str, bool]:
    """
    Check if specified files exist in the given folder.

    Args:
        folder_path (str): The path to the folder.
        filenames (list[str]): List of filenames to check.

    Returns:
        dict[str, bool]: A dictionary indicating the existence of each file.
    """
    files_exist = {}
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        files_exist[filename] = os.path.isfile(file_path)

    if len(files_exist) == 1:
        return files_exist[filenames[0]]
    else:
        return files_exist

def check_folder_exists(folder_path: str) -> bool:
    """
    Check if the specified folder exists.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    return os.path.isdir(folder_path)
