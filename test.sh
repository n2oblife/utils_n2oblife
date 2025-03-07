#!/bin/bash

# Check if pytest is installed
if python -m pytest --version &> temp.txt; then
    pytest_pst=$(awk '{print $1}' temp.txt)  # Extract first word
    rm temp.txt  # Delete temp file

    if [[ "$pytest_pst" == "pytest" ]]; then
        python -m pytest tests/  # Run pytest
    else
        install_pytest
    fi
else
    install_pytest
fi

# Function to install pytest and run tests
install_pytest() {
    python -m pip install --upgrade pytest
    python -m pytest tests/
}
