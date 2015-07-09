
Claduc: A portable high-level API with CUDA or OpenCL back-end
================

Claduc provides a C++ interface to the OpenCL API and/or CUDA API. This interface is high-level: all the details of setting up an OpenCL platform and device are handled automatically, as well as for example OpenCL and CUDA memory management. A similar high-level API is also provided by Khronos's `cl.hpp`, so why would someone use Claduc instead? The main reason is portability: Claduc provides two header files which both implement the exact same API, but with a different back-end. This allows __porting between OpenCL and CUDA by simply changing the header file!__

Claduc is written in C++11 and wraps CUDA and OpenCL objects in smart pointers, thus handling memory management automatically. It uses the CUDA driver API, since this is the closest to the OpenCL API, but it uses the OpenCL terminology, since this is the most generic. It compiles OpenCL and/or CUDA kernels at run-time, possible in CUDA only since release 7.0. Claduc handles the host API only: it still requires two versions of the kernel (although some simple defines could omit this requirement).


What does it look like?
-------------

To get started, include either of the two headers:

```c++
#include <clpp11.h>
// or:
#include <cupp11.h>
```

Here is a simple example of setting-up platform 0 and selecting device 2:

```c++
auto platform = Claduc::Platform(0);
auto device = Claduc::Device(platform, 2);
```

Next, we'll create a CUDA/OpenCL context and a queue (== CUDA stream) on this device:

```c++
auto context = Claduc::Context(device);
auto queue = Claduc::Queue(context, device);
```

And, once the context and queue are created, we can allocate and upload data to the device:

```c++
auto host_mem = std::vector<float>(size);
auto device_mem = Claduc::Buffer<float>(context, Claduc::BufferAccess::kReadWrite, size);
device_mem.WriteBuffer(queue, size, host_mem);
```

Further examples are included in the `samples` folder. To start with Claduc, check out `samples/simple.cc`, which shows how to compile and launch a simple kernel. The full [Claduc API reference](doc/api.md) is also available in the current repository.


Why would I use Claduc?
-------------

The main reasons to use Claduc are:

* __Portability__: the CUDA and OpenCL Claduc headers implement the exact same API.
* __Memory management__: smart pointers allocate and free memory automatically.
* __Error checking__: all CUDA and OpenCL API calls are automatically checked for errors.
* __Abstraction__: Claduc provides a higher-level interface than OpenCL, CUDA, and `cl.hpp`.
* __Easy to use__: simply ship two OS/hardware-independent header files, no compilation needed.
* __Low overhead__ : all function calls are automatically in-lined by the compiler.
* __Native compiler__: Claduc code can be compiled with a normal C++ compiler, there is no need to use `nvcc`.

Nevertheless, there are also several cases when Claduc is not suitable:

* When fine-grained control is desired: Claduc makes abstractions to certain OpenCL/CUDA handles and settings.
* When unsupported features are desired: only the most common cases are currently implemented. Although this is not a fundamental limitation, it is a practical one. For example, OpenGL interoperability and CUDA constant/texture memory are not supported.
* When run-time compilation is not an option: e.g. when compilation overhead is too high.

What are the pre-requisites?
-------------

The requirements to use the Claduc headers are:

* CUDA 7.0 or higher (for run-time compilation)
* OpenCL 1.1 or higher
* A C++11 compiler (e.g. GCC 4.7 or newer)

If you also want to compile the samples using the provided infrastructure, you'll also need:

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


FAQ
-------------

> Q: __After I include the Claduc CUDA header, the linker finds an undefined reference to `nvrtcGetErrorString'. What should I do?__
>
> A: You need to link against the NVIDIA Run-Time Compilation Library (NVRTC). For example, pass `-lnvrtc -L/opt/cuda/lib64` to the compiler.
