add_library(gale SHARED Gale.cu)
target_include_directories(gale
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(gale PUBLIC abstractTriangulation common)
target_link_libraries(gale PUBLIC VTK::CommonDataModel)
set_property(TARGET gale PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if (TTK_ENABLE_OPENMP)
    target_compile_options(gale PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
endif()