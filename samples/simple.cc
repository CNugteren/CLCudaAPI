
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a relatively simple toy example, in which an input vector is multiplied by
// a constant to produce an output vector. This example demonstrates the basic usage of the C++11
// interfaces to CUDA and OpenCL through CLCudaAPI.
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

// Runs with either OpenCL or CUDA as a back-end
#if USE_OPENCL
  #include "clpp11.h"
#else
  #include "cupp11.h"
#endif

// C++ includes
#include <vector>
#include <string>
#include <cstdio>

// =================================================================================================

// This example uses a single monolithic function
int main() {

  // Example CUDA/OpenCL program as a string. Note that the strings are loaded here as raw string
  // literals (using C++11's R"(string)" syntax). However, they can also be generated in-line or
  // perhaps placed in a separate file and loaded at run-time.
  #if USE_OPENCL
    auto program_string = R"(
    __kernel void multiply(__global float* x, __global float* y, const int factor) {
      const int tid = get_global_id(0);
      y[tid] = x[tid] * factor;
    })";
  #else
    auto program_string = R"(
    extern "C" __global__ void multiply(float* x, float* y, const int factor) {
      const int tid = threadIdx.x + blockDim.x*blockIdx.x;
      y[tid] = x[tid] * factor;
    })";
  #endif

  // ===============================================================================================

  // Sets the size of the vectors and the data-multiplication factor
  constexpr auto size = static_cast<size_t>(2048 * 2048);
  auto multiply_factor = 2;

  // Platform/device settings
  constexpr auto platform_id = size_t{0};
  constexpr auto device_id = size_t{0};

  // Initializes the CLCudaAPI platform and device. This initializes the OpenCL/CUDA back-end and
  // selects a specific device on the platform. The device class has methods to retrieve properties
  // such as the device name and vendor. More examples of device properties are given in the
  // `device_info.cc` sample program. 
  printf("\n## Initializing...\n");
  auto platform = CLCudaAPI::Platform(platform_id);
  auto device = CLCudaAPI::Device(platform, device_id);
  printf(" > Running on device '%s' of '%s'\n", device.Name().c_str(), device.Vendor().c_str());

  // Creates a new CLCudaAPI context and queue for this device. The queue can be used to schedule
  // commands such as launching a kernel or performing a device-host memory copy.
  auto context = CLCudaAPI::Context(device);
  auto queue = CLCudaAPI::Queue(context, device);

  // Creates a new CLCudaAPI event to be able to time kernels
  auto event = CLCudaAPI::Event();

  // Creates a new program based on the kernel string. Then, builds this program and checks for
  // any compilation errors. If there are any, they are printed and execution is halted.
  printf("## Compiling the kernel...\n");
  auto program = CLCudaAPI::Program(context, program_string);
  auto compiler_options = std::vector<std::string>{};
  auto build_status = program.Build(device, compiler_options);
  if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    auto message = program.GetBuildInfo(device);
    printf(" > Compiler error(s)/warning(s) found:\n%s\n", message.c_str());
    return 1;
  }

  // Populates regular host vectors with example data
  auto host_a = std::vector<float>(size);
  auto host_b = std::vector<float>(size);
  for (auto i=size_t{0}; i<host_a.size(); ++i) { host_a[i] = static_cast<float>(i); }
  for (auto &item: host_b) { item = 0.0f; }

  // Creates two new device buffers and copies the host data to these device buffers.
  auto dev_a = CLCudaAPI::Buffer<float>(context, queue, host_a.begin(), host_a.end());
  auto dev_b = CLCudaAPI::Buffer<float>(context, queue, host_b.begin(), host_b.end());

  // Creates the 'multiply' kernel from the compiled program and sets the three arguments. Note that
  // the indices of the arguments have to be set according to their order in the kernel.
  auto kernel = CLCudaAPI::Kernel(program, "multiply");
  kernel.SetArgument(0, dev_a);
  kernel.SetArgument(1, dev_b);
  kernel.SetArgument(2, multiply_factor);

  // Creates a 1-dimensional thread configuration with thread-blocks/work-groups of 256 threads
  // and a total number of threads equal to the number of elements in the input/output vectors.
  constexpr auto kWorkGroupSize = size_t{256};
  auto global = std::vector<size_t>{size};
  auto local = std::vector<size_t>{kWorkGroupSize};

  // Enqueues the kernel and waits for the result. Note that launching the kernel is always
  // a-synchronous and thus requires finishing the queue in order to complete the operation.
  printf("## Running the kernel...\n");
  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);
  printf(" > Took %.3lf ms\n", event.GetElapsedTime());

  // Reads the results back to the host memory
  dev_b.Read(queue, size, host_b);

  // Prints the results for a couple of indices to verify that the work has been done
  printf("## All done. Sampled verification:\n");
  const auto verification_indices = std::vector<size_t>{4, 900};
  for (const auto &index: verification_indices) {
    printf(" > %.lf*%d = %.lf\n", host_a[index], multiply_factor, host_b[index]);
  }

  // End of the example: no frees or clean-up needed
  return 0;
}

// =================================================================================================
