# file(GLOB TTK_SOURCES "*.cpp" "*.cu")

# add_library(ttk SHARED ${TTK_SOURCES})
# target_compile_features(ttk PUBLIC cxx_std_14)
# target_include_directories(ttk PUBLIC ${VTK_INCLUDE_DIRS})
# target_link_libraries(ttk PUBLIC ${VTK_LIBRARIES})
# set_property(TARGET ttk PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(ttk INTERFACE OpenMP::OpenMP_CXX)
# endif()

# set(TTK_BASECODES "")
file(GLOB BASECODE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(BASECODE ${BASECODE_DIRS})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${BASECODE})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${BASECODE})
    # list(APPEND TTK_BASECODES "${BASECODE}")
  endif()
endforeach()

# set(TTK_BASECODES ${TTK_BASECODES} CACHE INTERNAL "")
