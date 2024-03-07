add_library(skeleton SHARED "Skeleton.cpp" "Skeleton.h")
target_include_directories(skeleton
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(skeleton PUBLIC common)
target_link_libraries(skeleton PRIVATE Boost::boost)
