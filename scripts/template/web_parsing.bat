@if (@CodeSection == @Batch) @then
@echo off & setlocal

set "url=http://www.domain.com/data/page/1"


for /f "delims=" %%I in ('cscript /nologo /e:JScript "%~f0" "%url%"') do (
    rem // do something useful with %%I
    echo Link found: %%I
)

goto :EOF
@end // end batch / begin JScript hybrid code

// returns a DOM root object
function fetch(url) {
    var XHR = WSH.CreateObject("Microsoft.XMLHTTP"),
        DOM = WSH.CreateObject('htmlfile');

    XHR.open("GET",url,true);
    XHR.setRequestHeader('User-Agent','XMLHTTP/1.0');
    XHR.send('');
    while (XHR.readyState!=4) {WSH.Sleep(25)};
    DOM.write('<meta http-equiv="x-ua-compatible" content="IE=9" />');
    DOM.write(XHR.responseText);
    return DOM;
}

var DOM = fetch(WSH.Arguments(0)),
    links = DOM.getElementsByTagName('a');

for (var i in links)
    if (links[i].href && /\/post\/view\//i.test(links[i].href))
        WSH.Echo(links[i].href);