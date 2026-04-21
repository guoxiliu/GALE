cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(GALE LANGUAGES CXX CUDA)

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

option(TTK_ENABLE_KAMIKAZE "Enable Kamikaze compilation mode" ON)

option(TTK_ENABLE_OPENMP "Enable OpenMP support" TRUE)
if(TTK_ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        option(TTK_ENABLE_OMP_PRIORITY "Gives tasks priority, high perf improvement" OFF)
        if(OpenMP_CXX_VERSION_MAJOR GREATER_EQUAL 4 AND
            OpenMP_CXX_VERSION_MINOR GREATER_EQUAL 5)
            set(TTK_ENABLE_OMP_PRIORITY
                OFF
                CACHE
                BOOL
                "Enable priorities on opnemp tasks"
                FORCE
            )
        endif()
    else()
        message("OpenMP NOT FOUND!")
        if(TTK_ENABLE_OMP_PRIORITY)
            # priorities are only meaningful when openmp is on
            set(TTK_ENABLE_OMP_PRIORITY
                OFF
                CACHE
                BOOL
                "Enable priorities on opnemp tasks"
                FORCE
            )
        endif()
        return()
    endif()
endif()

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

find_package(Boost COMPONENTS system thread)
    if(NOT Boost_FOUND)
        find_package(Boost REQUIRED)
        if(Boost_FOUND)
            message(STATUS "BOOST_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
        endif()
endif()

message(STATUS "VTK_VERSION: ${VTK_VERSION}")
message("VTK_DIR: ${VTK_DIR}")
# message("VTK_LIBRARIES: ${VTK_LIBRARIES}")

# set(CMAKE_CUDA_ARCHITECTURES 60 61 75)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Adjusting CUDA FLAGS for debug build...")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -O0")
    #target_compile_options(all PRIVATE -g -lineinfo -O0)
    #target_compile_options(ALL_TARGET CUDA::CUDA_OPTIONS PRIVATE -G -O0)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0 -g -lineinfo")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -O0 -g -lineinfo")
endif()

# add_subdirectory("test")

# Source code
function(set_compile_options library)
    if (TTK_ENABLE_KAMIKAZE)
        target_compile_definitions(${library} PUBLIC TTK_ENABLE_KAMIKAZE)
    endif()

    if (TTK_ENABLE_OPENMP)
        target_compile_definitions(${library} PUBLIC TTK_ENABLE_OPENMP)
        target_link_libraries(${library} PUBLIC OpenMP::OpenMP_CXX)

        if (TTK_ENABLE_OMP_PRIORITY)
           target_compile_definitions(${library} PUBLIC TTK_ENABLE_OMP_PRIORITY)
        endif()
    endif()
endfunction()

add_subdirectory("source")

