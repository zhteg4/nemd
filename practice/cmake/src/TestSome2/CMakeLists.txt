cmake_minimum_required(VERSION 3.13)
project(testsome)
add_executable(testsome main.cpp)
add_subdirectory(glfw)
target_include_directories(${PROJECT_NAME} PUBLIC glfw/include/GLFW)
target_link_directories(testsome
    PRIVATE /usr/local/lib
    PRIVATE glfw/src)
target_link_libraries(testsome mearlymath glfw)
