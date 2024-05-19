#!/bin/bash

# Create a virtual environment
python3 -m venv ensq

# Activate the virtual environment
source ensq/bin/activate

# Install the required packages
pip install -r requirements.txt
