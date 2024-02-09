#include "../SomeLibDemo/cpp/adder.h"
#include <iostream>
#include <GLFW/glfw3.h>
#include <testsomeConfig.h>
int main(int argc, char* argv[]) {
  std::cout << mearlymath:: add(2,3) << '\n';
  std::cout << argv[0] << " version " << testsome_VERSION_MAJOR << "." << testsome_VERSION_MINOR << "\n";
  return 0;
 }
