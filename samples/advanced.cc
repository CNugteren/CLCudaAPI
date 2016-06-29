
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates more advanced usage of the C++11 interfaces to CUDA and OpenCL through
// CLCudaAPI. This includes 2D thread dimensions and asynchronous host-device communication. The
// example conserns a 2D convolution kernel with a very simple hard-coded 3x3 blur filter.
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

// Run with either OpenCL or CUDA as a back-end
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

  // This example passes different options to the run-time compiler based on which back-end is used
  #if USE_OPENCL
    auto compiler_options = std::vector<std::string>{};
  #else
    auto compiler_options = std::vector<std::string>{"--gpu-architecture=compute_35"};
  #endif

  // Example CUDA/OpenCL program as a string. Note that this is the first (header) part only, the
  // main body of the kernel is common among the two back-ends and is therefore not duplicated.
  #if USE_OPENCL
    auto program_head = R"(
    __kernel void convolution(__global float* x, __global float* y,
                              const int size_x, const int size_y) {
      const int tid_x = get_global_id(0);
      const int tid_y = get_global_id(1);
    )";
  #else
    auto program_head = R"(
    extern "C" __global__ void convolution(float* x, float* y,
                                           const int size_x, const int size_y) {
      const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
      const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
    )";
  #endif

  // The common body of the OpenCL/CUDA program. This is glued after the 'program_head' string.
  // It implements a star-based fixed 3x3 blur filter.
  auto program_tail = R"(
    float value = 0.0f;
    if (tid_x >= 1 && tid_y >= 1 && tid_x < size_x-1 && tid_y < size_y-1) {
      value += 0.2*x[(tid_y+1)*size_x + (tid_x  )];
      value += 0.2*x[(tid_y-1)*size_x + (tid_x  )];
      value += 0.2*x[(tid_y  )*size_x + (tid_x  )];
      value += 0.2*x[(tid_y  )*size_x + (tid_x+1)];
      value += 0.2*x[(tid_y  )*size_x + (tid_x-1)];
    }
    y[tid_y*size_x + tid_x] = value;
  })";
  auto program_string = std::string{program_head} + std::string{program_tail};

  // ===============================================================================================

  // Sets the size of the 2D input/output matrices
  constexpr auto size_x = size_t{2048};
  constexpr auto size_y = size_t{2048};
  auto size = size_x * size_y;

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

  // Creates a new program based on the kernel string. Note that the kernel string is moved-out when
  // constructing the program to save copying: it should no longer be used in the remainder of this
  // function.
  auto program = CLCudaAPI::Program(context, std::move(program_string));

  // Builds this program and checks for any compilation errors. If there are any, they are printed
  // and execution is halted.
  printf("## Compiling the kernel...\n");
  auto build_status = program.Build(device, compiler_options);
  if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    auto message = program.GetBuildInfo(device);
    printf(" > Compiler error(s)/warning(s) found:\n%s\n", message.c_str());
    return 1;
  }

  // Populate host matrices based on CUDA/OpenCL host buffers. When using the CUDA back-end, this
  // will create page-locked memories, benefiting from higher bandwidth when copying between the
  // host and device. These buffers mimic std::vector to some extend and can therefore be filled
  // using either the '[]' operator or range-based for-loops.
  auto host_a = CLCudaAPI::BufferHost<float>(context, size);
  auto host_b = CLCudaAPI::BufferHost<float>(context, size);
  for (auto x=size_t{0}; x<size_x; ++x) {
    for (auto y=size_t{0}; y<size_y; ++y) {
      host_a[y*size_x + x] = static_cast<float>(x + y/4);
    }
  }
  for (auto &item: host_b) { item = 0.0f; }

  // Creates two new device buffers and prints the sizes of these device buffers. Both buffers
  // in this example are readable and writable.
  printf("## Allocating device memory...\n");
  auto dev_a = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, size);
  auto dev_b = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, size);
  printf(" > Size of buffer A is %zu bytes\n", dev_a.GetSize());
  printf(" > Size of buffer B is %zu bytes\n", dev_b.GetSize());

  // Copies the matrices to the device a-synchronously. The queue is then finished to ensure that
  // the operations are completed before continuing.
  dev_a.WriteAsync(queue, size, host_a);
  dev_b.WriteAsync(queue, size, host_b);
  queue.Finish();

  // Creates the 'convolution' kernel from the compiled program and sets the four arguments. Note
  // that this uses the direct form instead of setting each argument separately.
  auto kernel = CLCudaAPI::Kernel(program, "convolution");
  auto size_x_int = static_cast<int>(size_x);
  auto size_y_int = static_cast<int>(size_y);
  kernel.SetArguments(dev_a, dev_b, size_x_int, size_y_int);

  // Creates a 2-dimensional thread configuration with thread-blocks/work-groups of 16x16 threads
  // and a total number of threads equal to the number of elements in the input/output matrices.
  constexpr auto kWorkGroupSizeX = size_t{16};
  constexpr auto kWorkGroupSizeY = size_t{16};
  auto global = std::vector<size_t>{static_cast<size_t>(size_x), static_cast<size_t>(size_y)};
  auto local = std::vector<size_t>{kWorkGroupSizeX, kWorkGroupSizeY};

  // Makes sure that the thread configuration is legal on this device
  if (!device.IsThreadConfigValid(local)) {
    printf("## Unsupported local thread configuration for this device, exiting.\n");
    return 1;
  }

  // Enqueues the kernel and waits for the result. Note that launching the kernel is always
  // a-synchronous and thus requires finishing the queue in order to complete the operation.
  printf("## Running the kernel...\n");
  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);
  printf(" > Took %.3lf ms\n", event.GetElapsedTime());

  // For illustration purposes, this copies the result into a new device buffer. The old result
  // buffer 'dev_b' is now no longer used.
  auto dev_b_copy = CLCudaAPI::Buffer<float>(context, CLCudaAPI::BufferAccess::kReadWrite, size);
  dev_b.CopyTo(queue, size, dev_b_copy);

  // Reads the results back from the new copy into the host memory
  dev_b_copy.ReadAsync(queue, size, host_b);
  queue.Finish();

  // Prints the results for a couple of indices to verify that the work has been done
  printf("## All done. Sampled verification:\n");
  const auto verification_indices = std::vector<size_t>{20};
  for (const auto &index: verification_indices) {
    printf(" > 0.2*%.lf + 0.2*%.lf + 0.2*%.lf + 0.2*%.lf + 0.2*%.lf = %.2lf\n",
           host_a[(index+1)*size_x + (index  )], host_a[(index-1)*size_x + (index  )],
           host_a[(index  )*size_x + (index  )], host_a[(index  )*size_x + (index+1)],
           host_a[(index  )*size_x + (index-1)],
           host_b[index*size_x + index]);
  }

  // End of the example: no frees or clean-up needed
  return 0;
}

// =================================================================================================
