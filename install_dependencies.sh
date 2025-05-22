#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing system dependencies..."
sudo apt install -y libcairo2 libcairo2-dev

echo "Installing Python packages..."
pip install cairosvg
pip install matplotlib
pip install accelerate
pip install wandb
pip install scikit-learn

echo "Installation complete."