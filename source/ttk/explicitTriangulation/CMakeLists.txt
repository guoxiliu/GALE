add_library(explicitTriangulation SHARED "ExplicitTriangulation.cpp" "ExplicitTriangulation.h")
target_include_directories(explicitTriangulation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(explicitTriangulation PUBLIC abstractTriangulation skeleton VTK::CommonDataModel)