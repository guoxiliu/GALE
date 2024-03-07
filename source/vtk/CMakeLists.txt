
add_executable(ttkTestTopoRelations ttkTestTopoRelations.cu)
set_property(TARGET ttkTestTopoRelations PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkTestTopoRelations PRIVATE gale testTopoRelations)
target_link_libraries(ttkTestTopoRelations PRIVATE OpenMP::OpenMP_CXX VTK::IOXML)
target_compile_options(ttkTestTopoRelations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

add_executable(ttkTestExplicit ttkTestExplicit.cpp)
target_link_libraries(ttkTestExplicit PRIVATE explicitTriangulation scalarFieldCriticalPoints testTopoRelations)
target_link_libraries(ttkTestExplicit PRIVATE OpenMP::OpenMP_CXX VTK::IOXML)

add_executable(ttkScalarFieldCriticalPoints ttkScalarFieldCriticalPoints.cu)
set_property(TARGET ttkScalarFieldCriticalPoints PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE gale scalarFieldCriticalPoints)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE OpenMP::OpenMP_CXX VTK::IOXML)
target_compile_options(ttkScalarFieldCriticalPoints PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE gale)
target_link_libraries(cudaBlockRelation PRIVATE VTK::IOXML)
