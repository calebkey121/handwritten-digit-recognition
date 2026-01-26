#!/bin/bash

# Check if environment already exists
if conda info --envs | grep -q "hdr"; then
    echo "Environment already exists. Activating..."
else
    # Create Conda environment
    conda create --name hdr
fi

# Activate Conda environment
conda activate hdr

# Install packages using pip
pip install -r requirements.txt