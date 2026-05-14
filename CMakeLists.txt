cmake_minimum_required(VERSION 4.1)
project(FractalHypercubeRC VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# ——— Static library: core pipeline (FractalReservoir, ESN, Readout) ———
add_library(FractalHypercubeRCCore STATIC
    ${PROJECT_SOURCE_DIR}/FractalReservoir.cpp
    ${PROJECT_SOURCE_DIR}/ESN.cpp
    ${PROJECT_SOURCE_DIR}/Readout.cpp
)

# ——— HypercubeCNN dependency (CNN readout) ———
# Sibling project at ../HypercubeCNN — https://github.com/dliptak001/HypercubeCNN
# Built as a static library. Readout.cpp #includes "HCNN.h"; the link target
# supplies hcnn::HCNN symbols.
set(HCNN_DIR "${PROJECT_SOURCE_DIR}/../HypercubeCNN")
set(HCNN_BUILD_DIR "${HCNN_DIR}/cmake-build-release")

add_library(HypercubeCNNCore STATIC IMPORTED)
set_target_properties(HypercubeCNNCore PROPERTIES
    IMPORTED_LOCATION "${HCNN_BUILD_DIR}/libHypercubeCNNCore.a"
    INTERFACE_INCLUDE_DIRECTORIES "${HCNN_DIR}"
)

# Include directories: PROJECT_SOURCE_DIR for in-tree builds.
target_include_directories(FractalHypercubeRCCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_link_libraries(FractalHypercubeRCCore PUBLIC HypercubeCNNCore)

# ——— Optimization and warning flags (library) ———
# Note: -flto is unavailable here — CLion 2024.3.2's bundled MinGW gcc 13.1
# was built without LTO support.  Would need a different MinGW distribution.
target_compile_options(FractalHypercubeRCCore PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
)

if(MINGW)
    target_link_options(FractalHypercubeRCCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Executables: only when building standalone (not via FetchContent) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(FractalHypercubeRC main.cpp)
    add_executable(BasicPrediction examples/BasicPrediction.cpp)
    add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp)
    add_executable(SignalClassification examples/SignalClassification.cpp)

    set(EXE_TARGETS FractalHypercubeRC BasicPrediction StreamingAnomaly SignalClassification)

    foreach(tgt IN LISTS EXE_TARGETS)
        target_link_libraries(${tgt} PRIVATE FractalHypercubeRCCore)
    endforeach()

    foreach(tgt IN LISTS EXE_TARGETS)
        target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
                $<$<CONFIG:Debug>:-O0 -g>
                -Wall -Wextra -Wno-unknown-pragmas
        )

        if(MINGW)
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
            # Workaround for a gcc 13.1 MinGW bug where
            # `std::type_info::operator==` in <typeinfo> is emitted out-of-line
            # at -O0 (Debug) and collides with libstdc++.a(tinfo.o). Release
            # (-O3) inlines it and is unaffected. Fixed upstream in gcc 13.2,
            # but CLion 2024.3.2 / 2026.1 both still bundle 13.1. Debug-only
            # so we don't mask real multi-def bugs in Release.
            target_link_options(${tgt} PRIVATE
                $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
        endif()
    endforeach()
endif()
