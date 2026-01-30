#!/bin/bash
set -e

APP_NAME="LiverSurgerySim"
APP_VERSION="1.2.0"
BUILD_DIR="build_appimage"
APPDIR="${BUILD_DIR}/AppDir"

echo "=========================================="
echo " AppImage Builder for ${APP_NAME}"
echo "=========================================="

# ビルド準備
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# CMakeビルド
echo "[1/4] Building..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -DUSE_GEOGRAM=ON -DENABLE_AVX2=OFF
make -j$(nproc)

# 実行ファイルを探す
if [ -f "Release/bin/${APP_NAME}" ]; then
    EXE="Release/bin/${APP_NAME}"
else
    EXE="bin/${APP_NAME}"
fi

# AppDir作成
echo "[2/4] Creating AppDir..."
mkdir -p ${APPDIR}/build/Release/bin
mkdir -p ${APPDIR}/usr/lib

cp ${EXE} ${APPDIR}/build/Release/bin/
chmod +x ${APPDIR}/build/Release/bin/${APP_NAME}

# リソースコピー
echo "[3/4] Copying resources..."
for d in model shader shaderGPUDuo shaders data; do
    [ -d "../$d" ] && cp -r "../$d" ${APPDIR}/
done

# デスクトップファイル（AppDirルートに！）
cat > ${APPDIR}/${APP_NAME}.desktop << EOF
[Desktop Entry]
Type=Application
Name=Liver Surgery Simulator
Exec=AppRun
Icon=${APP_NAME}
Categories=Science;
Terminal=false
EOF

# アイコン
cat > ${APPDIR}/${APP_NAME}.svg << 'ICONEOF'
<?xml version="1.0"?><svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><rect width="256" height="256" fill="#8B0000" rx="40"/><text x="128" y="170" font-size="140" text-anchor="middle" fill="white">肝</text></svg>
ICONEOF
ln -sf ${APP_NAME}.svg ${APPDIR}/.DirIcon

# AppRun
cat > ${APPDIR}/AppRun << 'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
cd "${HERE}"
exec "${HERE}/build/Release/bin/LiverSurgerySim" "$@"
APPRUN
chmod +x ${APPDIR}/AppRun

# ライブラリ収集
echo "[4/4] Creating AppImage..."
ldd ${APPDIR}/build/Release/bin/${APP_NAME} | grep "=> /" | awk '{print $3}' | while read lib; do
    case "$lib" in */libc.so*|*/libm.so*|*/libpthread.so*|*/libdl.so*|*/ld-linux*) ;; *)
        cp -n "$lib" ${APPDIR}/usr/lib/ 2>/dev/null || true ;;
    esac
done
[ -f /home/meidaikasai/geogram/build/lib/libgeogram.so ] && cp /home/meidaikasai/geogram/build/lib/libgeogram.so* ${APPDIR}/usr/lib/ 2>/dev/null || true

# appimagetool
cd ..
[ ! -f appimagetool ] && wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -O appimagetool && chmod +x appimagetool

ARCH=x86_64 ./appimagetool --appimage-extract-and-run ${BUILD_DIR}/AppDir ${APP_NAME}-${APP_VERSION}-x86_64.AppImage

echo ""
echo "=========================================="
echo " SUCCESS!"
echo " AppImage: ${APP_NAME}-${APP_VERSION}-x86_64.AppImage"
echo "=========================================="
