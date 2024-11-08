@echo off
chcp 65001
echo 啟動虛擬環境...
call .venv\Scripts\activate
if errorlevel 1 (
    echo 虛擬環境啟動失敗
    pause
    exit /b 1
)

echo 設定 Intel oneAPI 環境變數...

:: 設定 PyTorch GPU 環境
call "C:\Program Files (x86)\Intel\oneAPI\pytorch-gpu-dev-0.5\oneapi-vars.bat"
if errorlevel 1 (
    echo PyTorch GPU 環境設定失敗
    pause
    exit /b 1
)

:: 設定 OCLOC 環境
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\2024.2\env\vars.bat"
if errorlevel 1 (
    echo OCLOC 環境設定失敗
    pause
    exit /b 1
)

:: 設定 KMP_DUPLICATE_LIB_OK 環境變量
set KMP_DUPLICATE_LIB_OK=TRUE

echo 環境變數設定完成，開始執行 Python 應用...
:: 在新的命令提示字元視窗中啟動瀏覽器
start cmd /c "timeout /t 2 /nobreak && start http://localhost:5000/"

:: 在當前視窗執行 Python 應用（不使用 start）
python app.py

if errorlevel 1 (
    echo 程式執行失敗
    pause
    exit /b 1
)

@REM start python app.py

@REM :: 等待應用啟動
@REM timeout /t 2 /nobreak

@REM :: 打開默認瀏覽器
@REM start http://localhost:5000/

@REM if errorlevel 1 (
@REM     echo 程式執行失敗
@REM     pause
@REM     exit /b 1
@REM )

@REM echo 程式執行完成
@REM pause