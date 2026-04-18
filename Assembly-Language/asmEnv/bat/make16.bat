@echo off
REM make16.bat
REM By: hengchen@xjtu

REM Compile and Link 16-bit ASM program in dosbox with MASM50.
REM 
REM Command-line options (unless otherwise noted, they are case-sensitive):
REM Usage:
REM   make16 filename [d|debug|c|cv|codeview]
REM Description: 
REM   Compile, Link and debug 16-bit ASM program in dosbox with MASM50.
REM Options:
REM   filename: is the asm source filename, NOT include the .asm suffix
REM   d|debug:   After successful compile and link, call debug program
REM   c|cv|codeview: After successful compile and link, call codeview 
REM 
REM Compile Options:
REM   /c      Generate cross-reference
REM   /l[a]   Generate listing, a-list all
REM   /Zi	  Generate symbolic information for CodeView
REM
REM Link Options:
REM   /CODEVIEW  Generate symbolic information for CodeView
REM

REM ************* The following lines can be customized:
SET MASMPATH=C:\MASM50\bin\
SET INCLUDE=C:\MASM50\include
SET LIB=C:\MASM50\lib
REM **************************** End of customized lines

REM help
if '%1' == ''      goto usage
if '%1' == '/h'    goto usage
if '%1' == '/H'    goto usage
if '%1' == '/?'    goto usage
if '%1' == '/help' goto usage
if '%1' == '/HELP' goto usage

REM check the existence of filename.asm
if not exist %1.asm goto FileNotExist

REM Delete all old files related to %1.asm
if exist %1.lib del %1.lib
if exist %1.map del %1.map
if exist %1.exe del %1.exe
if exist %1.crf del %1.crf
if exist %1.lst del %1.lst
if exist %1.obj del %1.obj

REM Invoke MASM.EXE (the assembler masm 5.0):
%MASMPATH%masm /c /l /Zi %1.asm %1.obj %1.lst %1.crf

REM check error after compile phase
if errorlevel 1 goto terminate

REM After successful compile, Link it
%MASMPATH%link %1.obj /CODEVIEW, %1.exe, %1.map,, 

REM check error after link phase
if errorlevel 1 goto terminate

echo     Generate %1.exe successfully.
echo.

REM check the second argument
REM Only one argument
if '%2' == ''      goto terminate

REM check whether it is debug/d with case insensitive
if '%2' == 'd'     goto NEED_DEBUG
if '%2' == 'D'     goto NEED_DEBUG
if '%2' == 'debug' goto NEED_DEBUG
if '%2' == 'DEBUG' goto NEED_DEBUG

REM check whether it is c/cv/codeview with case insensitive
if '%2' == 'c'         goto NEED_CODEVIEW
if '%2' == 'C'         goto NEED_CODEVIEW
if '%2' == 'cv'        goto NEED_CODEVIEW
if '%2' == 'CV'        goto NEED_CODEVIEW
if '%2' == 'codeview'  goto NEED_CODEVIEW
if '%2' == 'CODEVIEW'  goto NEED_CODEVIEW

REM Invalid argument
goto usage

REM debug the %1.exe 
:NEED_DEBUG
echo.
echo ==================================================================
echo     debug %1.exe ...
echo ==================================================================
echo.
%MASMPATH%debug %1.exe
goto terminate

REM codeview the %1.exe
:NEED_CODEVIEW
%MASMPATH%cv %1.exe
goto terminate

:FileNotExist
echo.
echo Error. Assemble file "%1.asm" does NOT exist.
echo.

:usage
echo ==================================================================
echo Usage:
echo   make16.bat [/h /? /help] filename [d debug c cv codeview]
echo.
echo Description
echo   Compile and Link Assembly code with MASM 5.0
echo.
echo Options:
echo   /h, /?, /help:    Show this help information
echo   filename:         Required, does NOT include .asm suffix
echo   d, debug:         Optional, call debug after link successfully
echo   c, cv, codeview:  Optional, call codeview after link successfully
echo.
echo example: Assume you want to compile hello.asm
echo   make16.bat hello 
echo ====================================================================

:terminate
