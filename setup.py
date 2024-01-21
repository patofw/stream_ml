# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# packages required (replaces requirements.txt)
required = [
    'setuptools-rust==1.8.1',
    'setuptools==66.0.0',
    'numpy==1.26.3',
    'scikit-learn==1.2.2',
    'pandas==2.2.0',
    'river==0.21.0',
    'dabl==0.2.5',
    'matplotlib',
    'nltk==3.8.1',
    'seaborn==0.12.2',
    'textblob==0.17.1',
    'textstat==0.7.3',
    "Flask==3.0.1"
]

setup(
    name="stream_ml",
    version="0.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
)
