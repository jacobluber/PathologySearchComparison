#!/bin/bash

# Check if a directory argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Ensure the provided argument is a directory
if [ ! -d "$1" ]; then
    echo "Error: The provided path is not a directory."
    exit 1
fi

# Search and delete .ipynb_checkpoints directories
find "$1" -type d -name .ipynb_checkpoints -exec rm -rf {} +

echo "Done deleting .ipynb_checkpoints directories."

