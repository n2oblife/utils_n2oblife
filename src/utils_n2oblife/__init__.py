### AUTO IMPORTS ###
# This file is automatically import every other file of the folder as a module
# To add new scripts, create a new file in this directory.
# only import this file as in your main to get everything at once


import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals())
del module

__version__='0.0.2'