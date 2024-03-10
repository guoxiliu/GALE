cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(AcTopo-GPU LANGUAGES CXX CUDA)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(VTK PATHS ~/usr/local)
find_package(VTK)

if(NOT VTK_FOUND)
    message("VTK NOT FOUND!")
    return()
endif()

option(TTK_ENABLE_OPENMP "Enable OpenMP support" TRUE)
if(TTK_ENABLE_OPENMP)
    find_package(OpenMP)
    if (NOT OpenMP_CXX_FOUND) 
        message("OpenMP NOT FOUND!")
        return()
    endif()

    find_package(Boost COMPONENTS system)
        if(NOT Boost_FOUND)
            find_package(Boost REQUIRED)
            if(Boost_FOUND)
                message(STATUS "BOOST_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
            endif()
    endif()
endif()

message(STATUS "VTK_VERSION: ${VTK_VERSION}")
message("VTK_DIR: ${VTK_DIR}")
# message("VTK_LIBRARIES: ${VTK_LIBRARIES}")

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75)
set(CMAKE_CXX_STANDARD 14)

# add_subdirectory("test")

# Source code
add_subdirectory("source")