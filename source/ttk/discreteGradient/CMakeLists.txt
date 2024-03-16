add_library(discreteGradient SHARED DiscreteGradient.cpp)
target_include_directories(discreteGradient
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(discreteGradient PUBLIC abstractTriangulation common ftmTree)
set_compile_options(discreteGradient)