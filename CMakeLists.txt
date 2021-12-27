cmake_minimum_required(VERSION 3.17)
project(Filters)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 14)

add_executable(Filters main.cpp)
target_link_libraries(Filters ${OpenCV_LIBS})