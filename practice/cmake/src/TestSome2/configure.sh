#! /bin/sh

cmake -DGLFW_BUILD_DOCS=OFF -DUSE_GLFW=ON -S . -B out/build
