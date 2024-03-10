add_library(scalarFieldCriticalPoints SHARED ScalarFieldCriticalPoints.cpp)
target_include_directories(scalarFieldCriticalPoints
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(scalarFieldCriticalPoints PUBLIC abstractTriangulation common)
target_link_libraries(scalarFieldCriticalPoints PUBLIC )
