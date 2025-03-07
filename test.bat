@echo off
setlocal enabledelayedexpansion

python -m pytest --version > temp.txt

rem Extract first word from the temp file
for /f "usebackq" %%a in ( temp.txt ) do (
set pytest_pst=%%a
goto done
)

:done
    del temp.txt
    if /i "%pytest_pst%"=="pytest" (call :pytest) else (call :install_pytest)
    goto :eof

:install_pytest
    python -m pip install --upgrade pytest
    goto pytest

:pytest
    python -m pytest tests\
    goto :eof