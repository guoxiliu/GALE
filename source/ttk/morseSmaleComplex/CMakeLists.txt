add_library(morseSmaleComplex SHARED MorseSmaleComplex.cpp)
target_include_directories(morseSmaleComplex
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(morseSmaleComplex PUBLIC morseSmaleComplex2D morseSmaleComplex3D)
set_compile_options(morseSmaleComplex)