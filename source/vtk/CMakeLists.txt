# test explicit triangulation executable
add_executable(ttkTestExplicit ttkTestExplicit.cpp)
target_link_libraries(ttkTestExplicit PRIVATE explicitTriangulation scalarFieldCriticalPoints testTopoRelations)
target_link_libraries(ttkTestExplicit PRIVATE VTK::IOXML)

# test topological relations executable
add_executable(ttkTestTopoRelations ttkTestTopoRelations.cpp)
# set_property(TARGET ttkTestTopoRelations PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkTestTopoRelations PRIVATE gale testTopoRelations)
target_link_libraries(ttkTestTopoRelations PRIVATE VTK::IOXML)
set_compile_options(ttkTestTopoRelations)

# scalar field critical points executable
add_executable(ttkScalarFieldCriticalPoints ttkScalarFieldCriticalPoints.cpp)
# set_property(TARGET ttkScalarFieldCriticalPoints PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(ttkScalarFieldCriticalPoints PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE gale scalarFieldCriticalPoints)
target_link_libraries(ttkScalarFieldCriticalPoints PRIVATE VTK::IOXML)
set_compile_options(ttkScalarFieldCriticalPoints)

# ftm tree executable
add_executable(ttkFTMTree ttkFTMTree.cpp)
# # set_property(TARGET ttkDiscreteGradient PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkFTMTree PRIVATE gale ftmTree)
target_link_libraries(ttkFTMTree PRIVATE VTK::IOXML VTK::FiltersCore VTK::CommonExecutionModel)
# target_compile_options(ttkDiscreteGradient PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
set_compile_options(ttkFTMTree)

# discrete gradient executable
add_executable(ttkDiscreteGradient ttkDiscreteGradient.cpp)
# # set_property(TARGET ttkDiscreteGradient PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ttkDiscreteGradient PRIVATE gale discreteGradient)
target_link_libraries(ttkDiscreteGradient PRIVATE VTK::IOXML)
# target_compile_options(ttkDiscreteGradient PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
set_compile_options(ttkDiscreteGradient)

# test relation extraction with cuda
add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE gale)
target_link_libraries(cudaBlockRelation PRIVATE VTK::IOXML)
