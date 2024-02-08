cmake_minimum_required(VERSION 2.8)
project(mearlymath)
add_library(mearlymath adder.cpp)
set_target_properties(mearlymath PROPERTIES PUBLIC_HEADER adder.h)
install(TARGETS mearlymath LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include)

