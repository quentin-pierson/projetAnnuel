# cmake_minimum_required(VERSION <specify CMake version here>)
project(CppIA)

set(CMAKE_CXX_STANDARD 14)

add_library(CppIA SHARED library.cpp linearModel.cpp header.h structCollection.cpp arrayLib.cpp fonctionnalite.h mlp.cpp)
#add_executable(library library.cpp linearModel.cpp header.h structCollection.cpp arrayLib.cpp fonctionnalite.h mlp.cpp)

include_directories("${PROJECT_SOURCE_DIR}\\Eigen")
target_link_libraries(CppIA ${PROJECT_SOURCE_DIR}\\Eigen)

cmake_minimum_required(VERSION 3.19)