add_library(skeleton SHARED Skeleton.cpp)
target_include_directories(skeleton
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(skeleton PUBLIC common)
target_link_libraries(skeleton PRIVATE Boost::boost)
set_compile_options(skeleton)