#!/bin/sh

rm -rf ./cmake-build-*

mkdir -p ./cmake-build-debug/
mkdir -p ./cmake-build-release

cd ./cmake-build-debug/
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..

cd ../cmake-build-release
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
