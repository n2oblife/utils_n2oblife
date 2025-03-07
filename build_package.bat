@echo off
setlocal enabledelayedexpansion

:: Get the current folder name
for %%I in (.) do set "folder=%%~nxI"

:: Check if the folder is correct
if /I "%folder%"=="utils_n2oblife" (    
    :: Install the package
    echo Installing package...
    pip install -e .\

    :: Install dependencies
    echo Installing dependencies...
    pip install -r requirement.txt

    :: Build the package
    echo Building utils_n2oblife package...
    
    python -m build --version > temp.txt
    
    rem Extract first word from the temp file
    for /f "usebackq" %%a in ( temp.txt ) do (
    set pybuild_pst=%%a
    goto done
    )

    :done
        del temp.txt
        if /i "%pybuild_pst%"=="build" (call :pybuild) else (call :install_pybuild)
        goto :eof

    :install_pybuild
        python -m pip install --upgrade build
        goto pybuild

    :pybuild
        python -m build
        goto :eof
    
    echo %RED%Package built and installed.%RESET%
) else (
    echo Not in utils_n2oblife folder
)

endlocal
