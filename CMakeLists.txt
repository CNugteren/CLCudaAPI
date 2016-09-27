
# ==================================================================================================
# This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This provides a simple build infrastructure for the sample programs. The option USE_OPENCL can be
# used to toggle between a CUDA or OpenCL back-end.
#
# ==================================================================================================
#
# Copyright 2015 SURFsara
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==================================================================================================

# CMake project details
cmake_minimum_required(VERSION 2.8.10)
project("CLCudaAPI" CXX)
set(CLCudaAPI_VERSION_MAJOR 8)
set(CLCudaAPI_VERSION_MINOR 0)

# ==================================================================================================

# Enable tests
option(ENABLE_TESTS "Build test-suite" ON)

# Select between OpenCL and CUDA back-end
option(USE_OPENCL "Use OpenCL instead of CUDA" ON)
if(USE_OPENCL)
  message("-- Building samples with OpenCL")
  add_definitions(-DUSE_OPENCL)
else()
  message("-- Building samples with CUDA")
endif()

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif(MSVC)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ox")
else()
  set(FLAGS "-O3 -std=c++11 -Wall -Wno-comment")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Requires CUDA and OpenCL. The latter is found through the included "FindOpenCL.cmake".
if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
else()
  find_package(CUDA REQUIRED)
endif()

# ==================================================================================================

# Include directories: C++11 headers and OpenCL/CUDA includes
include_directories(${CLCudaAPI_SOURCE_DIR}/include)
if(USE_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIRS})
else()
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Link directories: CUDA toolkit
if(USE_OPENCL)

else()
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

# ==================================================================================================

# Adds the sample programs
set(SAMPLE_PROGRAMS device_info simple advanced smallest)
foreach(SAMPLE ${SAMPLE_PROGRAMS})
  add_executable(${SAMPLE} samples/${SAMPLE}.cc)
  if(USE_OPENCL)
    target_link_libraries(${SAMPLE} ${OPENCL_LIBRARIES})
  else()
    target_link_libraries(${SAMPLE} cuda nvrtc)
  endif()
  install(TARGETS ${SAMPLE} DESTINATION bin)
endforeach()

# ==================================================================================================

# Optional: Enable inclusion of the test-suite
if(ENABLE_TESTS)
  enable_testing()
  include_directories(${CLCudaAPI_SOURCE_DIR}/test)
  add_executable(unit_tests test/unit_tests.cc)
  if(USE_OPENCL)
    target_link_libraries(unit_tests ${OPENCL_LIBRARIES})
  else()
    target_link_libraries(unit_tests cuda nvrtc)
  endif()
  add_test(unit_tests unit_tests)
endif()

# ==================================================================================================
