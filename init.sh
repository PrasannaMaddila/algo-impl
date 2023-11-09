#!/bin/bash
# intialisation script to create the environment and 
# install dependencies as found in requirements.txt
# Author : Prasanna <siva-sri-prasanna@inrae.fr>
# Created: 03 May 2023
# Last Modified: 21 July, 2023

# Create virtual env
echo "Creating Virtual Environment ..."
virtualenv -p python3.8 .venv; 
source .venv/bin/activate;

# Install requirements
echo "Creating Virtual Environment ..."
pip install -r requirements.txt

echo "Installing current project as an editable package ..."
pip install -e .

# Installing pre-commit hooks for 
# automatic code linting.
echo "Configuring pre-commit hooks ..."
pre-commit install

