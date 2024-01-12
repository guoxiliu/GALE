cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(AcTopo-GPU LANGUAGES CXX CUDA)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# set(VTK_INCLUDE_DIRS /usr/local/include/paraview-5.8)
set(VTK_INCLUDE_DIRS ~/usr/local/include/vtk-9.0)

find_package(VTK PATHS /usr/local/)
find_package(VTK PATHS ~/usr/local)
find_package(VTK)

if (NOT VTK_FOUND)
    message("VTK NOT FOUND!")
    return()
endif()

message(STATUS "VTK_VERSION: ${VTK_VERSION}")
message("VTK_INCLUDE_DIRS: ${VTK_INCLUDE_DIRS}")
message("VTK_LIBRARIES: ${VTK_LIBRARIES}")

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75)
set(CMAKE_CXX_STANDARD 14)

# add_subdirectory("test")
add_subdirectory("source")
