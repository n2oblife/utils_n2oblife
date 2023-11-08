@if (@a==@b) @end /*

:: fetch.bat <url>
:: fetch a web page

@echo off
setlocal
if "%~1"=="" goto usage
echo "%~1" | findstr /i "https*://" >NUL || goto usage

set "URL=%~1"
for /f "delims=" %%I in ('cscript /nologo /e:jscript "%~f0" "%URL%"') do (
    rem process the HTML line-by-line
    echo(%%I
)
goto :EOF

:usage
echo Usage: %~nx0 URL
echo     for example: %~nx0 http://www.google.com/
echo;
echo The URL must be fully qualified, including the http:// or https://
goto :EOF

JScript */
var x=new ActiveXObject("Microsoft.XMLHTTP");
x.open("GET",WSH.Arguments(0),true);
x.setRequestHeader('User-Agent','XMLHTTP/1.0');
x.send('');
while (x.readyState!=4) {WSH.Sleep(50)};
WSH.Echo(x.responseText);