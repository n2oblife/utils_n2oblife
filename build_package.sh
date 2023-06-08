#!/usr/bin/env bash

result=${PWD##*/}           # to assign to a variable
result=${result:-/}         # to correct for the case where PWD=/
folder="${PWD##*/}"         # for use as shell input

if [ $folder == utils_n2oblife ]; then
    echo Building utils_n2oblife package
    python3 -m build
    echo Installing package in env
    pip install -e ./
    echo Pushing to git : https://github.com/n2oblife/utils_n2oblife
    git add .
    git commit -m "building commit"
    git push
    echo Package built and pushed 
else 
    echo Not in utils_n2oblife folder
fi