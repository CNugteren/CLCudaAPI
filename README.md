
CLCudaAPI: A portable high-level API with CUDA or OpenCL back-end
================

CLCudaAPI provides a C++ interface to the OpenCL API and/or CUDA API. This interface is high-level: all the details of setting up an OpenCL platform and device are handled automatically, as well as for example OpenCL and CUDA memory management. A similar high-level API is also provided by Khronos's `cl.hpp`, so why would someone use CLCudaAPI instead? The main reason is portability: CLCudaAPI provides two header files which both implement the exact same API, but with a different back-end. This allows __porting between OpenCL and CUDA by simply changing the header file!__

CLCudaAPI is written in C++11 and wraps CUDA and OpenCL objects in smart pointers, thus handling memory management automatically. It uses the CUDA driver API, since this is the closest to the OpenCL API, but it uses the OpenCL terminology, since this is the most generic. It compiles OpenCL and/or CUDA kernels at run-time, possible in CUDA only since release 7.0. CLCudaAPI handles the host API only: it still requires two versions of the kernel (although some simple defines could omit this requirement).


What does it look like?
-------------

To get started, include either of the two headers:

```c++
#include "clpp11.h"
// or:
#include "cupp11.h"
```

Here is a simple example of setting-up platform 0 and selecting device 2:

```c++
auto platform = CLCudaAPI::Platform(0);
auto device = CLCudaAPI::Device(platform, 2);
```

Next, we'll create a CUDA/OpenCL context and a queue (== CUDA stream) on this device:

```c++
auto context = CLCudaAPI::Context(device);
auto queue = CLCudaAPI::Queue(context, device);
```

And, once the context and queue are created, we can allocate and upload data to the device:

```c++
auto host_mem = std::vector<float>(size);
auto device_mem = CLCudaAPI::Buffer<float>(context, size);
device_mem.WriteBuffer(queue, size, host_mem);
```

Further examples are included in the `samples` folder. To start with CLCudaAPI, check out `samples/simple.cc`, which shows how to compile and launch a simple kernel. The full [CLCudaAPI API reference](doc/api.md) is also available in the current repository.


Why would I use CLCudaAPI?
-------------

The main reasons to use CLCudaAPI are:

* __Portability__: the CUDA and OpenCL CLCudaAPI headers implement the exact same API.
* __Memory management__: smart pointers allocate and free memory automatically.
* __Error checking__: all CUDA and OpenCL API calls are automatically checked for errors.
* __Abstraction__: CLCudaAPI provides a higher-level interface than OpenCL, CUDA, and `cl.hpp`.
* __Easy to use__: simply ship two OS/hardware-independent header files, no compilation needed.
* __Low overhead__ : all function calls are automatically in-lined by the compiler.
* __Native compiler__: CLCudaAPI code can be compiled with a normal C++ compiler, there is no need to use `nvcc`.

Nevertheless, there are also several cases when CLCudaAPI is not suitable:

* When fine-grained control is desired: CLCudaAPI makes abstractions to certain OpenCL/CUDA handles and settings.
* When unsupported features are desired: only the most common cases are currently implemented. Although this is not a fundamental limitation, it is a practical one. For example, OpenGL interoperability and CUDA constant/texture memory are not supported.
* When run-time compilation is not an option: e.g. when compilation overhead is too high.

What are the pre-requisites?
-------------

The requirements to use the CLCudaAPI headers are:

* CUDA 7.0 or higher
* OpenCL 1.1 or higher
* A C++11 compiler (e.g. GCC 4.7, Clang 3.3, MSVC 2015 or newer)

If you also want to compile the samples and tests using the provided infrastructure, you'll also need:

* CMake 2.8.10 or higher


How do I compile the included examples with CMake?
-------------

Use CMake to create an out-of-source build:

```bash
mkdir build
cd build
cmake -DUSE_OPENCL=ON ..
make
```

Replace `-DUSE_OPENCL=ON` with `-DUSE_OPENCL=OFF` to use CUDA instead of OpenCL as a back-end. After compilation, the `build` folder will contain a binary for each of the sample programs included in the `samples` subfolder.


How do I compile the included test-suite with CMake?
-------------

Compile the examples (see above) will also compile the tests (unless `-DENABLE_TESTS=OFF` is set). The tests will either use the OpenCL or CUDA back-end, similar to the samples. After compilation, the tests can be run using CTest or as follows:

```bash
./unit_tests
```


FAQ
-------------

> Q: __After I include the CLCudaAPI CUDA header, the linker finds an undefined reference to `nvrtcGetErrorString'. What should I do?__
>
> A: You need to link against the NVIDIA Run-Time Compilation Library (NVRTC). For example, pass `-lnvrtc -L/opt/cuda/lib64` to the compiler.
