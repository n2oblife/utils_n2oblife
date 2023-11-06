import pickle
import shutil


def copyfile(src, dst):
    shutil.copyfile(src, dst)

def pickle_write(file_path, obj):
    """
    Serialize an object to a provided file_path
    """

    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def pickle_read(file_path):
    """
    De-serialize an object from a provided file_path
    """

    with open(file_path, 'rb') as file:
        return pickle.load(file)