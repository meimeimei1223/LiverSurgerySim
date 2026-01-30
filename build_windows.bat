@echo off
REM ========================================
REM Windows Build Script for LiverSurgerySim
REM ========================================
REM
REM 使用方法:
REM   build_windows.bat [preset]
REM
REM プリセット:
REM   debug    - デバッグビルド
REM   release  - リリースビルド（デフォルト）
REM   static   - 静的リンクビルド
REM   ninja    - Ninjaビルド（高速）
REM
REM 例:
REM   build_windows.bat
REM   build_windows.bat release
REM   build_windows.bat static

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo  LiverSurgerySim Windows Build Script
echo ==========================================
echo.

REM ========================================
REM vcpkg チェック
REM ========================================
if not defined VCPKG_ROOT (
    echo [ERROR] VCPKG_ROOT environment variable is not set!
    echo.
    echo ========================================
    echo  vcpkg Setup Instructions
    echo ========================================
    echo.
    echo 1. Install vcpkg:
    echo    git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
    echo    cd C:\vcpkg
    echo    .\bootstrap-vcpkg.bat
    echo.
    echo 2. Set environment variable:
    echo    setx VCPKG_ROOT "C:\vcpkg"
    echo.
    echo 3. Restart this terminal and run again.
    echo.
    exit /b 1
)

echo [OK] VCPKG_ROOT = %VCPKG_ROOT%

REM CMakeの確認
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found in PATH
    echo Please install CMake from https://cmake.org/download/
    exit /b 1
)
echo [OK] CMake found

REM ========================================
REM プリセット選択
REM ========================================
set PRESET=%1
if "%PRESET%"=="" set PRESET=release

if /i "%PRESET%"=="debug" (
    set CMAKE_PRESET=windows-debug
    set BUILD_CONFIG=Debug
) else if /i "%PRESET%"=="release" (
    set CMAKE_PRESET=windows-release
    set BUILD_CONFIG=Release
) else if /i "%PRESET%"=="static" (
    set CMAKE_PRESET=windows-static
    set BUILD_CONFIG=Release
) else if /i "%PRESET%"=="ninja" (
    set CMAKE_PRESET=windows-ninja
    set BUILD_CONFIG=Release
) else (
    echo [ERROR] Unknown preset: %PRESET%
    echo.
    echo Available presets:
    echo   debug    - Debug build
    echo   release  - Release build ^(default^)
    echo   static   - Static link build
    echo   ninja    - Fast build with Ninja
    exit /b 1
)

echo.
echo [INFO] Preset:       %PRESET%
echo [INFO] CMake Preset: %CMAKE_PRESET%
echo [INFO] Config:       %BUILD_CONFIG%
echo.

REM ========================================
REM vcpkg で依存関係をインストール
REM ========================================
echo [1/4] Installing dependencies via vcpkg...
echo       This may take several minutes on first run...
echo.

if /i "%PRESET%"=="static" (
    set TRIPLET=x64-windows-static
) else (
    set TRIPLET=x64-windows
)

%VCPKG_ROOT%\vcpkg install --triplet %TRIPLET%
if errorlevel 1 (
    echo [ERROR] vcpkg install failed!
    exit /b 1
)

echo.
echo [OK] Dependencies installed

REM ========================================
REM CMake Configure
REM ========================================
echo.
echo [2/4] Configuring CMake with preset: %CMAKE_PRESET%
echo.

cmake --preset %CMAKE_PRESET%
if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    echo.
    echo Possible causes:
    echo - Missing Visual Studio 2022
    echo - VCPKG_ROOT not set correctly
    echo - Dependencies not installed
    exit /b 1
)

echo.
echo [OK] CMake configured successfully

REM ========================================
REM Build
REM ========================================
echo.
echo [3/4] Building...
echo.

cmake --build build/%CMAKE_PRESET% --config %BUILD_CONFIG% --parallel
if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [OK] Build completed

REM ========================================
REM Package
REM ========================================
echo.
echo [4/4] Creating distribution package...
echo.

set OUTPUT_DIR=dist\LiverSurgerySim-windows-x64
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"

REM 実行ファイルをコピー
echo Searching for executable...
set EXE_FOUND=0

REM パターン1: build/preset/Release/bin/
if exist "build\%CMAKE_PRESET%\%BUILD_CONFIG%\bin\LiverSurgerySim.exe" (
    copy "build\%CMAKE_PRESET%\%BUILD_CONFIG%\bin\LiverSurgerySim.exe" "%OUTPUT_DIR%\"
    set EXE_FOUND=1
    echo   Found: build\%CMAKE_PRESET%\%BUILD_CONFIG%\bin\LiverSurgerySim.exe
)

REM パターン2: build/preset/bin/Release/
if %EXE_FOUND%==0 if exist "build\%CMAKE_PRESET%\bin\%BUILD_CONFIG%\LiverSurgerySim.exe" (
    copy "build\%CMAKE_PRESET%\bin\%BUILD_CONFIG%\LiverSurgerySim.exe" "%OUTPUT_DIR%\"
    set EXE_FOUND=1
    echo   Found: build\%CMAKE_PRESET%\bin\%BUILD_CONFIG%\LiverSurgerySim.exe
)

REM パターン3: build/preset/bin/
if %EXE_FOUND%==0 if exist "build\%CMAKE_PRESET%\bin\LiverSurgerySim.exe" (
    copy "build\%CMAKE_PRESET%\bin\LiverSurgerySim.exe" "%OUTPUT_DIR%\"
    set EXE_FOUND=1
    echo   Found: build\%CMAKE_PRESET%\bin\LiverSurgerySim.exe
)

REM パターン4: 再帰検索
if %EXE_FOUND%==0 (
    echo   Searching recursively...
    for /r "build\%CMAKE_PRESET%" %%f in (LiverSurgerySim.exe) do (
        echo   Found: %%f
        copy "%%f" "%OUTPUT_DIR%\"
        set EXE_FOUND=1
        goto :found_exe
    )
)

if %EXE_FOUND%==0 (
    echo [ERROR] Could not find executable!
    exit /b 1
)
:found_exe

REM DLLをコピー（動的ビルドの場合のみ）
if /i not "%PRESET%"=="static" (
    echo Copying DLLs...
    
    REM vcpkgのDLLをコピー
    if exist "%VCPKG_ROOT%\installed\x64-windows\bin" (
        for %%f in (
            glfw3.dll
            glew32.dll
            glew32d.dll
        ) do (
            if exist "%VCPKG_ROOT%\installed\x64-windows\bin\%%f" (
                copy "%VCPKG_ROOT%\installed\x64-windows\bin\%%f" "%OUTPUT_DIR%\" >nul
                echo   Copied: %%f
            )
        )
    )
    
    REM ビルドディレクトリのDLLもコピー
    for /r "build\%CMAKE_PRESET%" %%f in (*.dll) do (
        copy "%%f" "%OUTPUT_DIR%\" 2>nul >nul
    )
)

REM ========================================
REM リソースをコピー（重要！）
REM ========================================
echo Copying resources...

REM model/ - 3Dモデルファイル
if exist "model" (
    echo   Copying model/...
    xcopy /E /I /Q "model" "%OUTPUT_DIR%\model"
)

REM shader/ - シェーダーファイル
if exist "shader" (
    echo   Copying shader/...
    xcopy /E /I /Q "shader" "%OUTPUT_DIR%\shader"
)

REM shaders/ - 追加シェーダー
if exist "shaders" (
    echo   Copying shaders/...
    xcopy /E /I /Q "shaders" "%OUTPUT_DIR%\shaders"
)

REM data/ - データファイル
if exist "data" (
    echo   Copying data/...
    xcopy /E /I /Q "data" "%OUTPUT_DIR%\data"
)

REM glm/ (ランタイムで必要な場合)
if exist "glm" (
    echo   Copying glm/...
    xcopy /E /I /Q "glm" "%OUTPUT_DIR%\glm"
)

REM バージョン情報
echo LiverSurgerySim Windows Build > "%OUTPUT_DIR%\VERSION.txt"
echo Build Date: %date% %time% >> "%OUTPUT_DIR%\VERSION.txt"
echo Build Type: %PRESET% >> "%OUTPUT_DIR%\VERSION.txt"
echo CMake Preset: %CMAKE_PRESET% >> "%OUTPUT_DIR%\VERSION.txt"

REM README
echo LiverSurgerySim - Liver Surgery Simulator > "%OUTPUT_DIR%\README.txt"
echo. >> "%OUTPUT_DIR%\README.txt"
echo 使い方: >> "%OUTPUT_DIR%\README.txt"
echo   LiverSurgerySim.exe をダブルクリックして起動 >> "%OUTPUT_DIR%\README.txt"
echo. >> "%OUTPUT_DIR%\README.txt"
echo 外部モデルを使う場合: >> "%OUTPUT_DIR%\README.txt"
echo   model/ フォルダ内のファイルを差し替えてください >> "%OUTPUT_DIR%\README.txt"

REM ========================================
REM ZIP作成（7-Zipがあれば）
REM ========================================
where 7z >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Creating ZIP archive...
    cd dist
    7z a -tzip "LiverSurgerySim-windows-x64.zip" "LiverSurgerySim-windows-x64" >nul
    cd ..
    echo   Created: dist\LiverSurgerySim-windows-x64.zip
)

REM ========================================
REM 完了
REM ========================================
echo.
echo ==========================================
echo  BUILD SUCCESSFUL!
echo ==========================================
echo.
echo Output directory: %OUTPUT_DIR%
echo.
dir "%OUTPUT_DIR%"
echo.
echo To run:
echo   cd %OUTPUT_DIR%
echo   LiverSurgerySim.exe
echo.
echo To distribute:
echo   Share the entire %OUTPUT_DIR% folder
echo   or dist\LiverSurgerySim-windows-x64.zip
echo.

endlocal
exit /b 0
