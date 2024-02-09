#! /bin/sh

cmake -DGLFW_BUILD_DOCS=OFF -DUSE_GLFW=OFF -S . -B out/build
