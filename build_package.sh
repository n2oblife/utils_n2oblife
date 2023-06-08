#!/usr/bin/env bash

result=${PWD##*/}           # to assign to a variable
result=${result:-/}         # to correct for the case where PWD=/
folder="${PWD##*/}"         # for use as shell input

if [ $folder == utils_n2oblife ]; then
    echo "$(tput bold)" Building utils_n2oblife package "$(tput sgr0)"
    python -m build
    echo "$(tput bold)" Installing package in env "$(tput sgr0)"
    pip install -e ./
    echo "$(tput bold)" Pushing to git : "$(tput sgr0)" https://github.com/n2oblife/utils_n2oblife
    git add .
    git commit -m "building commit"
    git push
    echo "$(tput bold)" Package built and pushed  "$(tput sgr0)"
else 
    echo Not in utils_n2oblife folder
fi