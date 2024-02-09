#include "../SomeLibDemo/cpp/adder.h"
#include <iostream>
#include <testsomeConfig.h>
#ifdef USE_GLFW
    #include <GLFW/glfw3.h>
#endif
int main(int argc, char** argv) {
  std::cout << mearlymath:: add(2,3) << '\n';
  std::cout << argv[0] << " version " << testsome_VERSION_MAJOR << "." << testsome_VERSION_MINOR << "\n";
#ifdef USE_GLFW
  std::cout << "using glfw" << " " << glfwInit() << "\n";
#else
  std::cout << "not using glfw" << "\n";
#endif
  return 0;
 }
