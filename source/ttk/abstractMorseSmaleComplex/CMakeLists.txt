add_library(abstractMorseSmaleComplex SHARED AbstractMorseSmaleComplex.cpp)
target_include_directories(abstractMorseSmaleComplex
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(abstractMorseSmaleComplex PUBLIC abstractTriangulation discreteGradient)
set_compile_options(abstractMorseSmaleComplex)