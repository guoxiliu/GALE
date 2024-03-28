add_library(common SHARED BaseClass.cpp Debug.cpp Os.cpp)
target_include_directories(common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
set_compile_options(common)