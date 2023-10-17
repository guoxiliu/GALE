add_library(AcTopo INTERFACE)

include_directories(${VTK_INCLUDE_DIRS})

add_subdirectory("ttk")

add_executable(testExplicit MACOSX_BUNDLE testExplicit.cpp)
target_include_directories(testExplicit PRIVATE "utils/")
target_link_libraries(testExplicit PRIVATE ttk)


set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75)
set(CMAKE_CXX_STANDARD 14)

add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE ttk ${VTK_LIBRARIES})

# add_executable(cudaSparse cudaSparse.cu)
# set_property(TARGET cudaSparse PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cudaSparse PRIVATE ttk ${VTK_LIBRARIES})


# add_executable(cudaTest cudaTest.cu)
# set_property(TARGET cudaTest PROPERTY CUDA_SEPARABLE_COMPILATION ON)
