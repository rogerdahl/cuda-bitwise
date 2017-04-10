cmake_minimum_required(VERSION 3.5)
project(cuda-bitwise)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBRARIES ${CMAKE_SOURCE_DIR}/libraries/linux)
else ()
  set(LIBRARIES ${CMAKE_SOURCE_DIR}/libraries/win64)
endif ()

SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "Building DEBUG version")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
else()
  message(WARNING "Building RELEASE version")
endif()

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wfatal-errors -Wno-deprecated-declarations"
)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wfatal-errors -Wall -fopenmp -Wno-array-bounds")
# clang++-3.5 -std=c++11 -O3 -o bin/bitwise-clang main.cpp -I libraries/linux/cppformat-2.1.0/ -I /usr/include/ -fopenmp -Llibraries/linux/cppformat-2.1.0/cppformat -lcppformat
#

# fmt
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(FMT_LIBRARIES fmt)
  set(FMT_INCLUDE_DIR ${LIBRARIES}/fmt-3.0.1)
  set(FMT_LINK_DIR ${LIBRARIES}/fmt-3.0.1/fmt)
else ()
  set(FMT_LIBRARIES fmt.lib)
  set(FMT_INCLUDE_DIR ${LIBRARIES}/fmt-3.0.1)
  set(FMT_LINK_DIR ${LIBRARIES}/fmt-3.0.1/builds/fmt/Release)
  #set(FMT_LINK_DIR ${LIBRARIES}/fmt-3.0.1/builds/fmt/Debug)
endif ()

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SRC_DIR}
  ${FMT_INCLUDE_DIR}
)

link_directories(
  ${FMT_LINK_DIR}
)

link_libraries(
  ${CUDA_LIBRARIES}
  ${FMT_LIBRARIES}
)

cuda_add_executable(
  bitwise
  ${SRC_DIR}/bitwise-cuda/main.cu
  ${SRC_DIR}/bitwise-cuda/cuda_helpers.cpp
)

message(WARNING "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(WARNING "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
