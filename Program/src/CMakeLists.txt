cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MiniProject1)

find_package(OpenCV REQUIRED)

add_executable(MiniProject1 main.cpp image4_1.cpp helpFunctions.cpp helpFunctions.h image4_1.h image4_2.cpp image4_2.h)

target_link_libraries(MiniProject1 ${OpenCV_LIBS})

