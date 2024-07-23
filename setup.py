import utils_n2oblife
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirement.txt", "r") as req:
    requirement_text = req.read()

setup(
    name="utils_n2oblife",
    version=utils_n2oblife.__version__, 
    author="n2oblife", 
    author_email="zackanit@gmail.com", 
    description="My own package with useful functions", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/n2oblife/utils_n2oblife",
    packages=find_packages(exclude=['*.tests.*', 'waiting_features']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirement_text,
)
