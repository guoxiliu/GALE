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
target_link_libraries(ttkFTMTree PRIVATE gale ftmTree)
target_link_libraries(ttkFTMTree PRIVATE VTK::IOXML VTK::FiltersCore VTK::CommonExecutionModel)
set_compile_options(ttkFTMTree)

# discrete gradient executable
add_executable(ttkDiscreteGradient ttkDiscreteGradient.cpp)
target_link_libraries(ttkDiscreteGradient PRIVATE gale discreteGradient)
target_link_libraries(ttkDiscreteGradient PRIVATE VTK::IOXML)
set_compile_options(ttkDiscreteGradient)

# Morse smale complex executable
add_executable(ttkMorseSmaleComplex ttkMorseSmaleComplex.cpp)
target_link_libraries(ttkMorseSmaleComplex PRIVATE gale morseSmaleComplex)
target_link_libraries(ttkMorseSmaleComplex PRIVATE VTK::IOXML)
set_compile_options(ttkMorseSmaleComplex)

# test relation extraction with cuda
add_executable(cudaBlockRelation cudaBlockRelation.cu)
set_property(TARGET cudaBlockRelation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaBlockRelation PRIVATE gale)
target_link_libraries(cudaBlockRelation PRIVATE VTK::IOXML)

# test GPU critical points kernel with GALE method
add_executable(criticalPointsKernel criticalPointsKernel.cu gpu_kernel.cu)
# set_property(TARGET criticalPointsKernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(criticalPointsKernel PRIVATE gale testTopoRelations)
target_link_libraries(criticalPointsKernel PRIVATE VTK::IOXML)
# set_compile_options(criticalPointsKernel)
