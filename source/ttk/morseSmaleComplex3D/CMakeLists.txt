add_library(morseSmaleComplex3D SHARED MorseSmaleComplex3D.cpp)
target_include_directories(morseSmaleComplex3D
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(morseSmaleComplex3D PUBLIC abstractMorseSmaleComplex)
set_compile_options(morseSmaleComplex3D)