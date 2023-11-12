file(GLOB TTK_SOURCES "*.cpp" "*.cu")

add_library(ttk SHARED ${TTK_SOURCES})
target_compile_features(ttk PUBLIC cxx_std_14)
target_include_directories(ttk PUBLIC ${VTK_INCLUDE_DIRS})
target_link_libraries(ttk PUBLIC ${VTK_LIBRARIES})
set_property(TARGET ttk PROPERTY CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ttk INTERFACE OpenMP::OpenMP_CXX)
endif()
