add_library(abstractTriangulation SHARED AbstractTriangulation.cpp)
target_include_directories(abstractTriangulation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(abstractTriangulation PUBLIC common geometry)