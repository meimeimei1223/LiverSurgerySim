cmake_minimum_required(VERSION 3.16)
project(LiverSurgerySim VERSION 1.2.0)

# ========================================
# ビルドオプション
# ========================================
option(ENABLE_SSE2 "Enable SSE2 optimizations" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(USE_GEOGRAM "Enable GEOGRAM for Voronoi3D segmentation" ON)
option(BUILD_STATIC "Build with static linking" OFF)

# C++標準を指定（C++17以上が必要: filesystem使用）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================================
# プラットフォーム検出と設定
# ========================================
if(WIN32)
    message(STATUS "========================================")
    message(STATUS " Platform: Windows")
    message(STATUS "========================================")
    
    # Windows固有の定義
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    
    # 静的リンク設定
    if(BUILD_STATIC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        message(STATUS " Static linking: ON")
    endif()
    
    # Visual Studio用設定
    if(MSVC)
        add_compile_options(/W3)
        add_compile_options(/MP)
        add_compile_options(/utf-8)
        add_compile_options(/openmp)
    endif()
    
elseif(UNIX AND NOT APPLE)
    message(STATUS "========================================")
    message(STATUS " Platform: Linux")
    message(STATUS "========================================")
    
    add_compile_options(-Wno-volatile)
    
    if(BUILD_STATIC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
        message(STATUS " Static linking: ON")
    endif()
    
elseif(APPLE)
    message(STATUS "========================================")
    message(STATUS " Platform: macOS (experimental)")
    message(STATUS "========================================")
    add_compile_options(-Wno-volatile)
endif()

# ========================================
# 依存パッケージの検索
# ========================================

# --- OpenMP ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS " OpenMP: Found (${OpenMP_CXX_VERSION})")
else()
    message(WARNING " OpenMP: Not found - parallel processing will be slower")
endif()

# --- OpenGL ---
find_package(OpenGL REQUIRED)
message(STATUS " OpenGL: ${OPENGL_LIBRARIES}")

# --- GLEW ---
if(WIN32)
    find_package(GLEW CONFIG QUIET)
    if(NOT GLEW_FOUND)
        find_package(GLEW REQUIRED)
    endif()
    
    if(TARGET GLEW::GLEW)
        set(GLEW_TARGET GLEW::GLEW)
    elseif(TARGET GLEW::glew)
        set(GLEW_TARGET GLEW::glew)
    else()
        set(GLEW_TARGET ${GLEW_LIBRARIES})
    endif()
else()
    find_package(GLEW REQUIRED)
    set(GLEW_TARGET ${GLEW_LIBRARIES})
endif()
message(STATUS " GLEW: Found")

# --- GLFW ---
if(WIN32)
    find_package(glfw3 CONFIG QUIET)
    if(NOT glfw3_FOUND)
        find_package(glfw3 REQUIRED)
    endif()
    set(GLFW_TARGET glfw)
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW QUIET glfw3)
    endif()
    
    if(NOT GLFW_FOUND)
        find_package(glfw3 REQUIRED)
        set(GLFW_TARGET glfw)
    else()
        set(GLFW_TARGET ${GLFW_LIBRARIES})
    endif()
endif()
message(STATUS " GLFW: Found")

# ========================================
# GEOGRAM設定（オプション）
# ========================================
if(USE_GEOGRAM)
    if(WIN32)
        find_path(GEOGRAM_INCLUDE_DIR geogram/basic/common.h
            HINTS 
                $ENV{GEOGRAM_ROOT}/include
                ${CMAKE_SOURCE_DIR}/third_party/geogram/include
                "C:/geogram/include"
        )
        find_library(GEOGRAM_LIBRARY 
            NAMES geogram
            HINTS 
                $ENV{GEOGRAM_ROOT}/lib
                ${CMAKE_SOURCE_DIR}/third_party/geogram/lib
                "C:/geogram/lib"
        )
    else()
        set(GEOGRAM_ROOT_CANDIDATES
            $ENV{GEOGRAM_ROOT}
            "/home/$ENV{USER}/geogram"
            "/home/meidaikasai/geogram"
            "/usr/local"
        )
        
        foreach(ROOT ${GEOGRAM_ROOT_CANDIDATES})
            if(EXISTS "${ROOT}/build/lib/libgeogram.so")
                set(GEOGRAM_LIBRARY "${ROOT}/build/lib/libgeogram.so")
                set(GEOGRAM_INCLUDE_DIR "${ROOT}/src/lib")
                break()
            endif()
        endforeach()
    endif()
    
    if(GEOGRAM_LIBRARY AND EXISTS "${GEOGRAM_LIBRARY}")
        message(STATUS "========================================")
        message(STATUS " GEOGRAM: ENABLED")
        message(STATUS "   Library: ${GEOGRAM_LIBRARY}")
        message(STATUS "   Include: ${GEOGRAM_INCLUDE_DIR}")
        message(STATUS "========================================")
        add_definitions(-DGEOGRAM_AVAILABLE=1)
        set(GEOGRAM_FOUND TRUE)
    else()
        message(WARNING " GEOGRAM: Not found - Voronoi3D disabled")
        add_definitions(-DGEOGRAM_AVAILABLE=0)
        set(GEOGRAM_FOUND FALSE)
        set(GEOGRAM_LIBRARY "")
        set(GEOGRAM_INCLUDE_DIR "")
    endif()
else()
    message(STATUS " GEOGRAM: Disabled by user")
    add_definitions(-DGEOGRAM_AVAILABLE=0)
    set(GEOGRAM_FOUND FALSE)
endif()

# ========================================
# ソースファイル（src/ ディレクトリ）
# ========================================
file(GLOB SOURCES 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

file(GLOB HEADERS
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)

# tetmesh_processor_standalone.cpp を除外（存在する場合）
list(FILTER SOURCES EXCLUDE REGEX ".*tetmesh_processor_standalone\\.cpp$")

# ソースファイル数を表示
list(LENGTH SOURCES SOURCE_COUNT)
list(LENGTH HEADERS HEADER_COUNT)
message(STATUS " Source files: ${SOURCE_COUNT}")
message(STATUS " Header files: ${HEADER_COUNT}")

# 出力ディレクトリ（build/Release/bin または build/Debug/bin）
if(CMAKE_BUILD_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release/bin)
endif()

# ========================================
# 実行ファイル作成
# ========================================
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# ========================================
# インクルードディレクトリ
# ========================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/stb_image
    ${CMAKE_SOURCE_DIR}/glm
    ${GLEW_INCLUDE_DIRS}
)

if(GEOGRAM_FOUND AND GEOGRAM_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GEOGRAM_INCLUDE_DIR})
endif()

# ========================================
# ライブラリリンク
# ========================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENGL_LIBRARIES}
    ${GLEW_TARGET}
    ${GLFW_TARGET}
)

if(GEOGRAM_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GEOGRAM_LIBRARY})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Windows固有のシステムライブラリ
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        opengl32
        gdi32
        user32
        shell32
    )
endif()

# ========================================
# 最適化フラグ
# ========================================
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE 
            /O2
            /GL
            /fp:fast
        )
        target_link_options(${PROJECT_NAME} PRIVATE /LTCG)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -O3 
            -DNDEBUG 
            -ffast-math
        )
        
        if(ENABLE_AVX2)
            target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -mfma)
        elseif(ENABLE_SSE2)
            target_compile_options(${PROJECT_NAME} PRIVATE -msse2)
        endif()
    endif()
endif()

# ========================================
# インストール設定
# ========================================
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

if(EXISTS "${CMAKE_SOURCE_DIR}/data")
    install(DIRECTORY data/ DESTINATION share/${PROJECT_NAME}/data)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/model")
    install(DIRECTORY model/ DESTINATION share/${PROJECT_NAME}/model)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/shader")
    install(DIRECTORY shader/ DESTINATION share/${PROJECT_NAME}/shader)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/shaderGPUDuo")
    install(DIRECTORY shaderGPUDuo/ DESTINATION share/${PROJECT_NAME}/shaderGPUDuo)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/shaders")
    install(DIRECTORY shaders/ DESTINATION share/${PROJECT_NAME}/shaders)
endif()

# ========================================
# ビルド情報サマリー
# ========================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS " Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Project:       ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Generator:     ${CMAKE_GENERATOR}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Sources:       ${SOURCE_COUNT} files in src/")
message(STATUS "  GEOGRAM:       ${GEOGRAM_FOUND}")
message(STATUS "  OpenMP:        ${OpenMP_CXX_FOUND}")
message(STATUS "  Static Build:  ${BUILD_STATIC}")
if(ENABLE_AVX2)
    message(STATUS "  SIMD:          AVX2")
elseif(ENABLE_SSE2)
    message(STATUS "  SIMD:          SSE2")
endif()
message(STATUS "========================================")
message(STATUS "")
