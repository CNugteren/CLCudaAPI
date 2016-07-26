
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a compact OpenCL/CUDA example inspired by the 'quest for the smallest OpenCL
// program': http://arrayfire.com/quest-for-the-smallest-opencl-program/
//
// =================================================================================================
//
// Copyright 2015 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

// Compile using OpenCL ...
#if USE_OPENCL
  #include "clpp11.h"
  static auto program_string = R"(
  __kernel void add(__global const float* a, __global const float* b, __global float* c) {
    unsigned idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
  })";

// ... or use CUDA instead
#else
  #include "cupp11.h"
  static auto program_string = R"(
  extern "C" __global__ void add(const float* a, const float* b, float* c) {
    unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
    c[idx] = a[idx] + b[idx];
  })";
#endif

#include <cstdio>

int main() {
  constexpr auto platform_id = size_t{0};
  constexpr auto device_id = size_t{0};
  auto platform = CLCudaAPI::Platform(platform_id);
  auto device = CLCudaAPI::Device(platform, device_id);
  auto context = CLCudaAPI::Context(device);
  auto queue = CLCudaAPI::Queue(context, device);
  auto event = CLCudaAPI::Event();

  // Creates and populates device memory
  constexpr auto elements = size_t{1024};
  auto data = std::vector<float>(elements, 5);
  auto a = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, elements);
  auto b = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, elements);
  auto c = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, elements);
  a.Write(queue, elements, data);
  b.Write(queue, elements, data);

  // Compiles and launches the kernel
  auto program = CLCudaAPI::Program(context, program_string);
  auto compiler_options = std::vector<std::string>{};
  program.Build(device, compiler_options);
  auto kernel = CLCudaAPI::Kernel(program, "add");
  kernel.SetArguments(a, b, c);
  kernel.Launch(queue, {elements}, {128}, event.pointer());
  queue.Finish(event);

  // Reads the results back to the host memory
  auto result = std::vector<float>(elements, 0);
  c.Read(queue, elements, result);
  for (auto &r: result) { printf("%.lf ", r); }
  printf("\n");
  return 0;
}

// =================================================================================================
