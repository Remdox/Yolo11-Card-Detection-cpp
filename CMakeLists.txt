cmake_minimum_required(VERSION 3.10)
project(ProjectCV)
add_compile_options(-Wall -g)

# If you move the onnxruntime directory away from externals, you can set the path from here:
# set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/path/...")
# otherwise, either external/onnxruntime-x64-1.21.0 is used as fallback:
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/external/onnxruntime-x64-1.21.0")
    message(STATUS "WARNING: ONNXRUNTIME_DIR not defined. Falling back to external/ folder :) \n")
endif()
# or, if you don't have external but you have onnxruntime installed on the system, it will be correctly included by cmake.


list(APPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_DIR}/lib64/cmake/onnxruntime")
list(APPEND CMAKE_PREFIX_PATH "/usr/local")
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED CONFIG)

# Shared
add_library(shared src/shared.cpp)
target_include_directories(shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shared PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

# 1ST LIB
add_library(marco_annunziata src/marco_annunziata.cpp)
target_include_directories(marco_annunziata PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marco_annunziata PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

#2ND LIB
add_library(hermann_serain src/hermann_serain.cpp)
target_include_directories(hermann_serain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hermann_serain PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

#3RD LIB
add_library(sveva_turola src/sveva_turola.cpp)
target_include_directories(sveva_turola PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sveva_turola PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

# EXE

add_executable(finalProject src/finalProject.cpp)
target_link_libraries(finalProject
    marco_annunziata
    hermann_serain
    sveva_turola
    shared #Lasciare shared per ultimo
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)
