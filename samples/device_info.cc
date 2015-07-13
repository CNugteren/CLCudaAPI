
// =================================================================================================
// This file is part of the Claduc project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a generic version of 'clinfo' (OpenCL) and 'deviceQuery' (CUDA). This
// demonstrates some of the features of Claduc's generic Device class.
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
  #include <clpp11.h>
#else
  #include <cupp11.h>
#endif

// C++ includes
#include <vector>
#include <string>
#include <cstdio>

// =================================================================================================

// Example implementation of a device-query/info program
int main() {

  // Platform/device settings
  constexpr auto platform_id = 0;
  constexpr auto device_id = 0;

  // Initializes the Claduc platform and device. This initializes the OpenCL/CUDA back-end and
  // selects a specific device on the platform.
  auto platform = Claduc::Platform(platform_id);
  auto device = Claduc::Device(platform, device_id);

  // Prints information about the chosen device. Most of these results should stay the same when
  // switching between the CUDA and OpenCL back-ends.
  printf("\n## Printing device information...\n");
  printf(" > Platform ID                  %d\n", platform_id);
  printf(" > Device ID                    %d\n", device_id);
  printf(" > Framework version            %s\n", device.Version().c_str());
  printf(" > Vendor                       %s\n", device.Vendor().c_str());
  printf(" > Device name                  %s\n", device.Name().c_str());
  printf(" > Device type                  %s\n", device.Type().c_str());
  printf(" > Max work-group size          %lu\n", device.MaxWorkGroupSize());
  printf(" > Max thread dimensions        %lu\n", device.MaxWorkItemDimensions());
  printf(" > Max work-group sizes:\n");
  for (auto i=size_t{0}; i<device.MaxWorkItemDimensions(); ++i) {
    printf("   - in the %lu-dimension         %lu\n", i, device.MaxWorkItemSizes()[i]);
  }
  printf(" > Local memory per work-group  %lu bytes\n", device.LocalMemSize());
  printf(" > Device capabilities          %s\n", device.Capabilities().c_str());
  printf(" > Core clock rate              %lu MHz\n", device.CoreClock());
  printf(" > Number of compute units      %lu\n", device.ComputeUnits());
  printf(" > Allocatable memory size      %lu bytes\n", device.MemorySize());
  printf(" > Memory clock rate            %lu MHz\n", device.MemoryClock());
  printf(" > Memory bus width             %lu bits\n", device.MemoryBusWidth());

  // End of the example: no frees or clean-up needed
  return 0;
}

// =================================================================================================
