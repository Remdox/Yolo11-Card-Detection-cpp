cmake_minimum_required(VERSION 3.10)
project(ProjectCV)
add_compile_options(-Wall -g)

find_package(OpenCV REQUIRED)

# Shared
add_library(shared src/shared.cpp)
target_include_directories(shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shared PUBLIC ${OpenCV_LIBS})

# 1ST LIB
add_library(marco_annunziata src/marco_annunziata.cpp)
target_include_directories(marco_annunziata PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marco_annunziata PUBLIC ${OpenCV_LIBS})

#2ND LIB
add_library(hermann_serain src/hermann_serain.cpp)
target_include_directories(hermann_serain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hermann_serain PUBLIC ${OpenCV_LIBS})

#3RD LIB
add_library(sveva_turola src/sveva_turola.cpp)
target_include_directories(sveva_turola PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sveva_turola PUBLIC ${OpenCV_LIBS})

# EXE

add_executable(finalProject src/finalProject.cpp)
target_link_libraries(finalProject
    marco_annunziata
    hermann_serain
    sveva_turola
    shared #Lasciare shared per ultimo
    ${OpenCV_LIBS}
)
