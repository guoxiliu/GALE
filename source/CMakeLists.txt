add_library(AcTopo INTERFACE)

include_directories(${VTK_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory("ttk")

add_executable(ttkTestTopoRelations ttkTestTopoRelations.cu)
set_property(TARGET ttkTestTopoRelations PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkTestTopoRelations PRIVATE ttk)

add_executable(ttkScalarFieldCriticalPoints ttkScalarFieldCriticalPoints.cu)
set_property(TARGET ttkScalarFieldCriticalPoints PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE ttk)

add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE ttk)

# add_executable(cudaSparse cudaSparse.cu)
# set_property(TARGET cudaSparse PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cudaSparse PRIVATE ttk ${VTK_LIBRARIES})


# add_executable(cudaTest cudaTest.cu)
# set_property(TARGET cudaTest PROPERTY CUDA_SEPARABLE_COMPILATION ON)
