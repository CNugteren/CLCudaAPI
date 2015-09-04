
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

// =================================================================================================

// Tests all public methods of the Platform class
SCENARIO("platforms can be created and used", "[Platform]") {
  GIVEN("An example platform") {
    auto platform = CLCudaAPI::Platform(kPlatformID);
    auto num_devices = platform.NumDevices();

    #if USE_OPENCL
    WHEN("its underlying data-structure is retrieved") {
      auto raw_platform = platform();

      THEN("a copy of this platform can be created") {
        auto platform_copy = CLCudaAPI::Platform(raw_platform);
        REQUIRE(platform_copy.NumDevices() == num_devices);
      }
    }
    #endif
  }
}

// =================================================================================================
