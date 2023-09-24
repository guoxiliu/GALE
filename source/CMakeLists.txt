add_library(AcTopo INTERFACE)

add_subdirectory("ttk")

add_executable(testExplicit MACOSX_BUNDLE testExplicit.cpp)
target_include_directories(testExplicit PRIVATE "utils/")
target_include_directories(testExplicit PUBLIC ${VTK_INCLUDE_DIRS})
target_link_libraries(testExplicit PRIVATE ttk)
