import os
import pprint
import importlib
import numpy as np

def pretty_print(name, input, val_width=40, key_width=0):
    """
    This function creates a formatted string from a given dictionary input.
    It may not support all data types, but can probably be extended.

    Args:
        name (str): name of the variable root
        input (dict): dictionary to print
        val_width (int): the width of the right hand side values
        key_width (int): the minimum key width, (always auto-defaults to the longest key!)

    Example:
        pretty_str = pretty_print('conf', conf.__dict__)
        pretty_str = pretty_print('conf', {'key1': 'example', 'key2': [1,2,3,4,5], 'key3': np.random.rand(4,4)})

        print(pretty_str)
        or
        logging.info(pretty_str)
    """

    # root
    pretty_str = name + ': {\n'

    # determine key width
    for key in input.keys(): key_width = max(key_width, len(str(key)) + 4)

    # cycle keys
    for key in input.keys():

        val = input[key]

        # round values to 3 decimals..
        if type(val) == np.ndarray: val = np.round(val, 3).tolist()

        # difficult formatting
        val_str = str(val)
        if len(val_str) > val_width:
            val_str = pprint.pformat(val, width=val_width, compact=True)
            val_str = val_str.replace('\n', '\n{tab}')
            tab = ('{0:' + str(4 + key_width) + '}').format('')
            val_str = val_str.replace('{tab}', tab)

        # more difficult formatting
        format_str = '{0:' + str(4) + '}{1:' + str(key_width) + '} {2:' + str(val_width) + '}\n'
        pretty_str += format_str.format('', key + ':', val_str)

    # close root object
    pretty_str += '}'

    return pretty_str

def split_file(file_path):
    """
    Lists a files parts such as base_path, file name and extension

    Example
        base, name, ext = file_parts('path/to/file/dog.jpg')
        print(base, name, ext) --> ('path/to/file/', 'dog', '.jpg')
    """

    base_path, tail = os.path.split(file_path)
    name, ext = os.path.splitext(tail)

    return base_path, name, ext

def absolute_import(file_path):
    """
    Imports a python module / file given its ABSOLUTE path.

    Args:
         file_path (str): absolute path to a python file to attempt to import
    """

    # module name
    _, name, _ = split_file(file_path)

    # load the spec and module
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module