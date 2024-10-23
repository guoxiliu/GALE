add_library(ftmTree SHARED FTMTree.cpp FTMTree_CT.cpp FTMTree_MT.cpp FTMSegmentation.cpp)
target_include_directories(ftmTree
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<INSTALL_INTERFACE:include/ttk/base>
)
target_link_libraries(ftmTree PUBLIC abstractTriangulation geometry Boost::boost)
set_compile_options(ftmTree)