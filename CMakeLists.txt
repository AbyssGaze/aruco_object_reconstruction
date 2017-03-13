cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

project(opencv_aruco_test)
#include(CheckCXXCompilerFlag)

#opencv
find_package(OpenCV REQUIRED)
#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

FILE(GLOB_RECURSE INC_ALL "src/*.h")

FILE(GLOB_RECURSE SRC_CPP "src/*.cpp")

add_executable(${PROJECT_NAME} ${SRC_CPP}  ${INC_ALL})
#add_executable(${PROJECT_NAME} "src/main.cpp")
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS})
