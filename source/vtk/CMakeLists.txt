
# test topological relations executable
add_executable(ttkTestTopoRelations ttkTestTopoRelations.cpp)
# set_property(TARGET ttkTestTopoRelations PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkTestTopoRelations PRIVATE gale testTopoRelations)
target_link_libraries(ttkTestTopoRelations PRIVATE VTK::IOXML)
# target_compile_options(ttkTestTopoRelations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

# test explicit triangulation executable
add_executable(ttkTestExplicit ttkTestExplicit.cpp)
target_link_libraries(ttkTestExplicit PRIVATE explicitTriangulation scalarFieldCriticalPoints testTopoRelations)
target_link_libraries(ttkTestExplicit PRIVATE VTK::IOXML)

# scalar field critical points executable
add_executable(ttkScalarFieldCriticalPoints ttkScalarFieldCriticalPoints.cpp)
# set_property(TARGET ttkScalarFieldCriticalPoints PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE gale scalarFieldCriticalPoints)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE VTK::IOXML)
# target_compile_options(ttkScalarFieldCriticalPoints PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

# test relation extraction with cuda
add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE gale)
target_link_libraries(cudaBlockRelation PRIVATE VTK::IOXML)
