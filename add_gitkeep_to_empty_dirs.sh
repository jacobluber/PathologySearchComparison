#!/bin/bash

# Check if directory argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Ensure the provided argument is a directory
if [ ! -d "$1" ]; then
    echo "Error: The provided path is not a directory."
    exit 1
fi

# Find all empty directories and touch a .gitkeep file inside them
find "$1" -type d -empty -exec touch {}/.gitkeep \;

echo "Done adding .gitkeep to empty directories."

