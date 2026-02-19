cmake_minimum_required(VERSION 3.10)
project(ProjectCV)
add_compile_options(-Wall -g)

# If you move the onnxruntime directory away from externals, you can set the path from here:
# set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/path/...")
# otherwise, either external/onnxruntime-x64-1.21.0 is used as fallback:
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/external")
    message(STATUS "WARNING: ONNXRUNTIME_DIR not defined. Falling back to external/ folder :) \n")
endif()
# or, if you don't have external but you have onnxruntime installed on the system, it will be correctly included by cmake.

set(ORT_GPU_DIR "${ONNXRUNTIME_DIR}/onnxruntime-linux-x64-gpu-1.21.0")
set(ORT_CPU_DIR "${ONNXRUNTIME_DIR}/onnxruntime-x64-1.21.0")

if(IS_DIRECTORY "${ORT_GPU_DIR}")
    set(ONNX_ROOT "${ORT_GPU_DIR}")
    set(ORT_LIB_NAME "lib")
    message(STATUS "Using ONNX GPU config")
else()
    set(ONNX_ROOT "${ORT_CPU_DIR}")
    set(ORT_LIB_NAME "lib64")
    message(STATUS "Using ONNX CPU config")
endif()

include_directories("${ONNX_ROOT}/include/")
include_directories("${ONNX_ROOT}/include/onnxruntime/")

find_library(ORT_LIB NAMES onnxruntime HINTS "${ONNX_ROOT}/${ORT_LIB_NAME}")
find_package(OpenCV REQUIRED)

if(ORT_LIB)
    message(STATUS "Using ONNX library: ${ORT_LIB}")
else()
    message(WARNING "ONNX library NOT FOUND!")
endif()

# Shared
add_library(shared src/shared.cpp)
target_include_directories(shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shared PUBLIC ${OpenCV_LIBS} ${ORT_LIB})

# 1ST LIB
add_library(marco_annunziata src/marco_annunziata.cpp)
target_include_directories(marco_annunziata PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marco_annunziata PUBLIC ${OpenCV_LIBS} ${ORT_LIB})

#2ND LIB
add_library(hermann_serain src/hermann_serain.cpp)
target_include_directories(hermann_serain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hermann_serain PUBLIC ${OpenCV_LIBS} ${ORT_LIB})

#3RD LIB
add_library(sveva_turola src/sveva_turola.cpp)
target_include_directories(sveva_turola PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sveva_turola PUBLIC ${OpenCV_LIBS} ${ORT_LIB})

# EXE

add_executable(finalProject src/finalProject.cpp)
target_link_libraries(finalProject
    marco_annunziata
    hermann_serain
    sveva_turola
    shared #Lasciare shared per ultimo
    ${OpenCV_LIBS}
    ${ORT_LIB}
)
