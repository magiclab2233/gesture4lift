@echo off
chcp 65001 >nul
title 无接触智慧电梯控制器
cd /d "%~dp0"

echo ==========================================
echo   无接触智慧电梯控制器
echo ==========================================
echo.

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境！
    echo.
    echo 请先在 PowerShell 中执行以下命令创建虚拟环境：
    echo    python -m venv venv
    echo    .\venv\Scripts\Activate.ps1
    echo    python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 检查主程序是否存在
if not exist "main.py" (
    echo [错误] 未找到 main.py！
    pause
    exit /b 1
)

echo [信息] 正在启动无接触智慧电梯控制器...
echo [信息] 按 Q 键退出程序
echo.

:: 使用虚拟环境中的 Python 运行主程序
"venv\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo [错误] 程序运行出错！
    pause
) else (
    echo.
    echo [信息] 程序已退出。
    timeout /t 2 >nul
)
