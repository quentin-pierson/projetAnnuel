# cmake_minimum_required(VERSION <specify CMake version here>)
project(CppIA)

set(CMAKE_CXX_STANDARD 14)

add_library(CppIA SHARED library.cpp linearModel.cpp header.h structCollection.cpp arrayLib.cpp fonctionnalite.h)

include_directories(./Eigen)
