@echo off
cls

set getopt_ParmCounter=1
set paramc=1
set DEBUG=1

set argc=0
for %%x in (%*) do Set /A argc+=1
echo Number of arguments: %argc%
echo %*&echo.

set _myvar=%*

rem Loop through all command line arguments one at a time
:varloop
set isparam=1
for /f "tokens=1*" %%a in ('echo %_myvar%') DO (
   set getopt_Parm=%%a
   set _myvar=%%b
   call :paramtype

   rem shift along arguments and rerun loop
   if NOT "%%b"=="" goto varloop
)
goto :eof

:paramtype
rem If first character starts with a - or / it must be an option
if /i "%getopt_Parm:~0,1%"=="-" call :option
if /i "%getopt_Parm:~0,1%"=="/" call :option 
if /i "%isparam%"=="1" call :param
goto :eof

:option
   set isparam=0
   rem Set the Equal Index to the position of the colon.  0 means none was found
   for /f %%j in ('findstring %getopt_Parm% :') do set getopt_EqIdx=%%j

   rem If the index is GE 0 then we must have a colon in the option.
   if /i "%getopt_EqIdx%"=="0" (call :nocolon) else (call :colon)
   goto :eof

      :colon
         rem set the OPTION value to the stuff to the right of the colon
         set /a getopt_ParmNameEnd=%getopt_EqIdx%-2
         call set getopt_ParmName=%%getopt_Parm:~1,%getopt_ParmNameEnd%%%
         call set getopt_ParmValue=%%getopt_Parm:~%getopt_EqIdx%%%
         set OPTION_%getopt_ParmName%=%getopt_ParmValue%
         goto :eof

      :nocolon
         rem This is a flag, so simply set the value to 1
         set getopt_ParmName=%getopt_Parm:~1%
         set getopt_ParmValue=1
         set OPTION_%getopt_ParmName%=%getopt_ParmValue%
         goto :eof

:param
   rem There was no / or - found, therefore this must be a paramater, not an option
   set PARAM_%getopt_ParmCounter%=%getopt_Parm%
   set PARAM_0=%getopt_ParmCounter%
   set /a getopt_ParmCounter+=1
   goto :eof