add_library(morseSmaleComplex2D SHARED MorseSmaleComplex2D.cpp)
target_include_directories(morseSmaleComplex2D
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(morseSmaleComplex2D PUBLIC abstractMorseSmaleComplex)
set_compile_options(morseSmaleComplex2D)