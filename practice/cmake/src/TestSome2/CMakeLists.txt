cmake_minimum_required(VERSION 3.13)
project(testsome VERSION 1.0)
configure_file(testsomeConfig.h.in testsomeConfig.h)
add_executable(testsome main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})
add_subdirectory(glfw)
target_include_directories(${PROJECT_NAME} PUBLIC glfw/include/GLFW)
target_link_directories(testsome
    PRIVATE /usr/local/lib
    PRIVATE glfw/src)
target_link_libraries(testsome mearlymath glfw)
