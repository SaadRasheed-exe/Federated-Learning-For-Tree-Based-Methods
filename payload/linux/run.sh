#!/bin/bash

# Check if the pyenv folder does not exist, then unzip pyenv.zip and delete the zip file
if [ ! -d "pyenv" ]; then
    mkdir pyenv
    tar -xzf pyenvlinux.tar.gz -C pyenv
    rm pyenvlinux.tar.gz
fi

# Activate the virtual environment
source pyenv/bin/activate

# Install dependencies
pyenv/bin/python -m pip install -r requirements.txt

# Run the Python application
pyenv/bin/python api/app.py

# Exit the script
exit
