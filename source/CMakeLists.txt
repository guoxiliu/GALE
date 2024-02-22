add_library(AcTopo INTERFACE)

include_directories(${VTK_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory("ttk")

add_executable(ttkTestTopoRelations ttkTestTopoRelations.cu)
set_property(TARGET ttkTestTopoRelations PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkTestTopoRelations PRIVATE ttk OpenMP::OpenMP_CXX)
target_compile_options(ttkTestTopoRelations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

add_executable(ttkTestExplicit ttkTestExplicit.cpp)
target_link_libraries(ttkTestExplicit PRIVATE ttk OpenMP::OpenMP_CXX)

add_executable(ttkScalarFieldCriticalPoints ttkScalarFieldCriticalPoints.cu)
set_property(TARGET ttkScalarFieldCriticalPoints PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE ttk)
target_compile_options(ttkScalarFieldCriticalPoints PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE ttk)

add_executable(cudaSparseMatrix cudaSparseMatrix.cu)
set_property(TARGET cudaSparseMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaSparseMatrix PRIVATE ttk cusparse)

# add_executable(cudaSparse cudaSparse.cu)
# set_property(TARGET cudaSparse PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cudaSparse PRIVATE ttk ${VTK_LIBRARIES})


# add_executable(cudaTest cudaTest.cu)
# set_property(TARGET cudaTest PROPERTY CUDA_SEPARABLE_COMPILATION ON)
