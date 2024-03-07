# add_library(AcTopo INTERFACE)

# include_directories(${VTK_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory("ttk")
add_subdirectory("vtk")
