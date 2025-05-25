#!/bin/bash
# install uv
if command -v uv &> /dev/null; then
    echo "uv is already installed."
else 
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv run inception_v3.py