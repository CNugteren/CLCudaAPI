
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements unit tests based on the Catch header-only test framework.
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

// Use Catch
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Runs with either OpenCL or CUDA as a back-end
#if USE_OPENCL
  #include <clpp11.h>
#else
  #include <cupp11.h>
#endif

// Settings
const size_t kPlatformID = 0;
const size_t kDeviceID = 0;
const size_t kBufferSize = 10;

// =================================================================================================

SCENARIO("events can be created and used", "[Event]") {
  GIVEN("An example event") {
    #if !USE_OPENCL
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    auto queue = CLCudaAPI::Queue(context, device);
    #endif
    auto event = CLCudaAPI::Event();

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_event = event();
      THEN("a copy of this event can be created") {
        auto event_copy = CLCudaAPI::Event(raw_event);
        REQUIRE(event_copy() == event());
      }
    }
    #else // Not available for the OpenCL version
    WHEN("its underlying data-structures are retrieved") {
      auto raw_start = event.start();
      auto raw_end = event.end();
      THEN("their underlying data-structures are not null") {
        REQUIRE(raw_start != nullptr);
        REQUIRE(raw_end != nullptr);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto event_copy = CLCudaAPI::Event(event);
      THEN("its underlying data-structure is unchanged") {
        #if USE_OPENCL
          REQUIRE(event_copy() == event());
        #else
          REQUIRE(event_copy.start() == event.start());
          REQUIRE(event_copy.end() == event.end());
        #endif
      }
    }

    // TODO: Not working if nothing is recorded
    //WHEN("the elapsed time is retrieved") {
    //  auto elapsed_time = event.GetElapsedTime();
    //  THEN("its value is valid") {
    //    REQUIRE(elapsed_time == elapsed_time);
    //  }
    //}
  }
}

// =================================================================================================

SCENARIO("platforms can be created and used", "[Platform]") {
  GIVEN("An example platform") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto num_devices = platform.NumDevices();

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_platform = platform();
      THEN("a copy of this platform can be created") {
        auto platform_copy = CLCudaAPI::Platform(raw_platform);
        REQUIRE(platform_copy.NumDevices() == num_devices);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto platform_copy = CLCudaAPI::Platform(platform);
      THEN("the platform's properties remain unchanged") {
        REQUIRE(platform_copy.NumDevices() == num_devices);
      }
    }
  }
}

// =================================================================================================

TEST_CASE("a list of all platforms can be retrieved", "[Platform]") {
  auto all_platforms = CLCudaAPI::GetAllPlatforms();
  REQUIRE(all_platforms.size() > 0);
  for (auto &platform : all_platforms) {
    auto num_devices = platform.NumDevices();
    REQUIRE(num_devices > 0);
  }
}

// =================================================================================================

SCENARIO("devices can be created and used", "[Device][Platform]") {
  GIVEN("An example device on a platform") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);

    GIVEN("...and device properties") {
      auto device_version = device.Version();
      auto device_vendor = device.Vendor();
      auto device_name = device.Name();
      auto device_type = device.Type();
      auto device_max_work_group_size = device.MaxWorkGroupSize();
      auto device_max_work_item_dimensions = device.MaxWorkItemDimensions();
      auto device_max_work_item_sizes = device.MaxWorkItemSizes();
      auto device_local_mem_size = device.LocalMemSize();
      auto device_capabilities = device.Capabilities();
      auto device_core_clock = device.CoreClock();
      auto device_compute_units = device.ComputeUnits();
      auto device_memory_size = device.MemorySize();
      auto device_max_alloc_size = device.MaxAllocSize();
      auto device_memory_clock = device.MemoryClock();
      auto device_memory_bus_width = device.MemoryBusWidth();

      // TODO: test for valid device properties

      WHEN("its underlying data-structure is retrieved") {
        auto raw_device = device();
        THEN("a copy of this device can be created") {
          auto device_copy = CLCudaAPI::Device(raw_device);
          REQUIRE(device_copy.Name() == device_name); // Only verifying device name
        }
      }

      WHEN("a copy is created using the copy constructor") {
        auto device_copy = CLCudaAPI::Device(device);
        THEN("the device's properties remain unchanged") {
          REQUIRE(device_copy.Name() == device_name); // Only verifying device name
        }
      }

      WHEN("the local memory size is tested") {
        THEN("the maximum local memory size should be considered valid") {
          REQUIRE(device.IsLocalMemoryValid(device_local_mem_size) == true);
        }
        THEN("more than the maximum local memory size should be considered invalid") {
          REQUIRE(device.IsLocalMemoryValid(device_local_mem_size+1) == false);
        }
      }

      WHEN("the local thread configuration is tested") {
        THEN("equal to the maximum size in one dimension should be considered valid") {
          REQUIRE(device.IsThreadConfigValid({device_max_work_item_sizes[0],1,1}) == true);
          REQUIRE(device.IsThreadConfigValid({1,device_max_work_item_sizes[1],1}) == true);
          REQUIRE(device.IsThreadConfigValid({1,1,device_max_work_item_sizes[2]}) == true);
        }
        THEN("more than the maximum size in one dimension should be considered invalid") {
          REQUIRE(device.IsThreadConfigValid({device_max_work_item_sizes[0]+1,1,1}) == false);
          REQUIRE(device.IsThreadConfigValid({1,device_max_work_item_sizes[1]+1,1}) == false);
          REQUIRE(device.IsThreadConfigValid({1,1,device_max_work_item_sizes[2]+1}) == false);
        }
      }
    }
  }
}

// =================================================================================================

SCENARIO("contexts can be created and used", "[Context][Device][Platform]") {
  GIVEN("An example context on a device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);

    WHEN("its underlying data-structure is retrieved") {
      auto raw_context = context();
      THEN("a copy of this context can be created") {
        auto context_copy = CLCudaAPI::Context(raw_context);
        REQUIRE(context_copy() != nullptr);
      }
    }

    WHEN("a copy is created using the copy constructor") {
      auto context_copy = CLCudaAPI::Context(context);
      THEN("its underlying data-structure is not null") {
        REQUIRE(context_copy() != nullptr);
      }
    }
  }
}

// =================================================================================================

SCENARIO("programs can be created and used", "[Program][Context][Device][Platform]") {
  GIVEN("An example program for a specific context and device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    #if USE_OPENCL
      auto source = R"(
      __kernel void add(__global const float* a, __global const float* b, __global float* c) {
        unsigned idx = get_global_id(0);
        c[idx] = a[idx] + b[idx];
      })";

    // ... or use CUDA instead
    #else
      auto source = R"(
      extern "C" __global__ void add(const float* a, const float* b, float* c) {
        unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
        c[idx] = a[idx] + b[idx];
      })";
    #endif
    auto options = std::vector<std::string>();

    auto program = CLCudaAPI::Program(context, source);
    auto build_result = program.Build(device, options);
    REQUIRE(build_result == CLCudaAPI::BuildStatus::kSuccess);

    WHEN("an compiled IR is generated from the compiled program") {
      auto ir = program.GetIR();
      THEN("a new program can be created based on the IR") {
        auto new_program = CLCudaAPI::Program(device, context, ir);
        auto new_build_result = new_program.Build(device, options);
        REQUIRE(new_build_result == CLCudaAPI::BuildStatus::kSuccess);
      }
    }
  }
}

// =================================================================================================

SCENARIO("queues can be created and used", "[Queue][Context][Device][Platform][Event]") {
  GIVEN("An example queue associated to a context and device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    auto queue = CLCudaAPI::Queue(context, device);

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_queue = queue();
      THEN("a copy of this queue can be created") {
        auto queue_copy = CLCudaAPI::Queue(raw_queue);
        REQUIRE(queue_copy() != nullptr);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto queue_copy = CLCudaAPI::Queue(queue);
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue_copy() != nullptr);
      }
    }

    WHEN("the associated context is retrieved") {
      auto context_copy = queue.GetContext();
      THEN("their underlying data-structures match") {
        REQUIRE(context_copy() == context());
      }
    }
    WHEN("the associated device is retrieved") {
      auto device_copy = queue.GetDevice();
      THEN("their underlying data-structures match") {
        REQUIRE(device_copy() == device());
      }
    }

    WHEN("the queue is synchronised") {
      queue.Finish();
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue() != nullptr);
      }
    }
    WHEN("the queue is synchronised using an event") {
      auto event = CLCudaAPI::Event();
      queue.Finish(event);
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue() != nullptr);
      }
    }
  }
}

// =================================================================================================

SCENARIO("host buffers can be created and used", "[BufferHost][Context][Device][Platform]") {
  GIVEN("An example host buffer for a specific context and device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    auto size = static_cast<size_t>(kBufferSize);
    auto buffer_host = CLCudaAPI::BufferHost<float>(context, size);

    // TODO: Fill in
  }
}

// =================================================================================================

SCENARIO("device buffers can be created and used", "[Buffer][Context][Device][Platform]") {
  GIVEN("An example device buffer for a specific context and device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    auto size = static_cast<size_t>(kBufferSize);
    auto buffer = CLCudaAPI::Buffer<float>(context, size);

    // TODO: Fill in
  }
}

// =================================================================================================

SCENARIO("kernels can be created and used", "[Kernel][Program][Context][Device][Platform]") {
  GIVEN("An example device buffer for a specific context and device") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto device = CLCudaAPI::Device(platform, kDeviceID);
    auto context = CLCudaAPI::Context(device);
    auto source = std::string{""};
    auto program = CLCudaAPI::Program(context, source);
    auto name = std::string{""};
    //auto kernel = CLCudaAPI::Kernel(program, name);

    // TODO: Fill in
  }
}

// =================================================================================================
