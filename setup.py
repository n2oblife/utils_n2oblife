from setuptools import setup, find_packages
import os
import own_utils

setup(name = 'own_utils', 
      version = own_utils.__version__ , 
      packages = find_packages(exclude=['*.tests.*', 'waiting_features']),
      # install_requires = ['os'],
	long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read())
      #  packages_dir = {'' : 'les_packages_maison'})