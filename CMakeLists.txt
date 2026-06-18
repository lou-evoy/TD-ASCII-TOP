# =============================================================================
# ASCII TOP — TouchDesigner Custom Operator (CUDA, edge-preserving ASCII mosaic)
# Validated: TouchDesigner 2025.32050 (TOP API v12), CUDA 13.3, VS 2022/2026, CMake >= 3.24.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(AsciiTOP LANGUAGES CXX CUDA)

set(TD_SDK_DIR
    "C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
    CACHE PATH "Folder containing TOP_CPlusPlusBase.h and CPlusPlus_Common.h")

option(AX_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc" ON)

if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
    message(FATAL_ERROR "TOP_CPlusPlusBase.h not found in TD_SDK_DIR='${TD_SDK_DIR}'.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(AX_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
    CACHE STRING "GPU architectures to embed (CMake CUDA_ARCHITECTURES syntax)")

find_package(CUDAToolkit REQUIRED)

add_library(AsciiTOP SHARED
    src/AsciiTOP.cpp
    src/AsciiCUDA.cu
    src/GlyphAtlas.cpp
)

target_include_directories(AsciiTOP PRIVATE
    "${TD_SDK_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(AsciiTOP PRIVATE CUDA::cudart_static)

set_target_properties(AsciiTOP PROPERTIES
    PREFIX ""
    OUTPUT_NAME "AsciiTOP"
    CUDA_ARCHITECTURES "${AX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Static
)

set_property(TARGET AsciiTOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

target_compile_definitions(AsciiTOP PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

target_compile_options(AsciiTOP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)
if(AX_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(AsciiTOP PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()

if(DEFINED ENV{USERPROFILE})
    set(TD_PLUGIN_DIR "$ENV{USERPROFILE}/Documents/Derivative/Plugins")
    add_custom_target(install_to_td
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TD_PLUGIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:AsciiTOP>" "${TD_PLUGIN_DIR}/"
        DEPENDS AsciiTOP
        COMMENT "Copying AsciiTOP.dll to ${TD_PLUGIN_DIR}")
endif()
