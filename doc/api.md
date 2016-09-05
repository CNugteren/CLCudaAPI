CLCudaAPI: API reference
================

This file describes the high-level API for both the CUDA and OpenCL back-end of the CLCudaAPI headers. On top of the described API, each class has a constructor which takes the regular OpenCL or CUDA data-type and transforms it into a CLCudaAPI class. Furthermore, each class also implements a `()` operator which returns the regular OpenCL or CUDA data-type.


CLCudaAPI::Event
-------------

Constructor(s):

* `Event()`:
Creates a new event, to be used for example when timing kernels.

Public method(s):

* `void WaitForCompletion() const`:
Waits for completion of an event (OpenCL) or does nothing (CUDA).

* `float GetElapsedTime() const`:
Retrieves the elapsed time in milliseconds of the last recorded event (e.g. a device kernel). This method first makes sure that the last event is finished before computing the elapsed time.


CLCudaAPI::Platform
-------------

Constructor(s):

* `Platform(const size_t platform_id)`:
When using the OpenCL back-end, this initializes a new OpenCL platform (e.g. AMD SDK, Intel SDK, NVIDIA SDK) specified by the integer `platform_id`. When using the CUDA back-end, this initializes the CUDA driver API. The `platform_id` argument is ignored: there is only one platform.

Non-member function(s):

* `std::vector<Platform> GetAllPlatforms()`:
Retrieves a vector containing all available platforms.


CLCudaAPI::Device
-------------

Constructor(s):

* `Device(const Platform &platform, const size_t device_id)`:
Initializes a new OpenCL or CUDA device on the specified platform. The `device_id` defines which device should be selected.

Public method(s):

* `std::string Version() const`:
Retrieves which version of the OpenCL standard is supported (OpenCL back-end) or which CUDA driver is used (CUDA back-end).

* `std::string Vendor() const`:
Retrieves the name of the vendor of the device.

* `std::string Name() const`:
Retrieves the name of the device.

* `std::string Type() const`:
Retrieves the type of the devices. Possible return values are 'CPU', 'GPU', 'accelerator', or 'default'.

* `size_t MaxWorkGroupSize() const`:
Retrieves the maximum total number of threads in an OpenCL work-group or CUDA thread-block.

* `size_t MaxWorkItemDimensions() const`:
Retrieves the maximum number of dimensions (e.g. 2D or 3D) in an OpenCL work-group or CUDA thread-block.

* `unsigned long LocalMemSize() const`:
Retrieves the maximum amount of on-chip scratchpad memory ('local memory') available to a single OpenCL work-group or CUDA thread-block.

* `std::string Capabilities() const`:
In case of the OpenCL back-end, this returns a list of the OpenCL extensions supported. For CUDA, this returns the device capability (e.g. SM 3.5).

* `size_t CoreClock() const`:
Retrieves the device's core clock frequency in MHz.

* `size_t ComputeUnits() const`:
Retrieves the number of compute units (OpenCL terminology) or multi-processors (CUDA terminology) in the device.

* `unsigned long MemorySize() const`:
Retrieves the total global memory size.

* `unsigned long MaxAllocSize() const`:
Retrieves the maximum amount of allocatable global memory per allocation.

* `size_t MemoryClock() const`:
Retrieves the device's memory clock frequency in MHz (CUDA back-end) or 0 (OpenCL back-end).

* `size_t MemoryBusWidth() const`:
Retrieves the device's memory bus-width in bits (CUDA back-end) or 0 (OpenCL back-end).

* `bool IsLocalMemoryValid(const size_t local_mem_usage) const`:
Given a requested amount of local on-chip scratchpad memory, this method returns whether or not this is a valid configuration for this particular device.

* `bool IsThreadConfigValid(const std::vector<size_t> &local) const`:
Given a requested OpenCL work-group or CUDA thread-block configuration `local`, this method returns whether or not this is a valid configuration for this particular device.

* `bool IsCPU() const`:
Determines whether this device is of the CPU type.

* `bool IsGPU() const`:
Determines whether this device is of the GPU type.

* `bool IsAMD() const`:
Determines whether this device is of the AMD brand.

* `bool IsNVIDIA() const`:
Determines whether this device is of the NVIDIA brand.

* `bool IsIntel() const`:
Determines whether this device is of the Intel brand.

* `bool IsARM() const`:
Determines whether this device is of the ARM brand.

CLCudaAPI::Context
-------------

Constructor(s):

* `Context(const Device &device)`:
Initializes a new context on a given device. On top of this context, CLCudaAPI can create new programs, queues and buffers.


CLCudaAPI::Program
-------------

Constants(s):

* `enum class BuildStatus { kSuccess, kError, kInvalid }`
Status of the run-time compiler. It can complete successfully (`kSuccess`), generated compiler warnings/errors (`kError`), or consider the source-code invalid (`kInvalid`).

Constructor(s):

* `Program(const Context &context, std::string source)`:
Creates a new OpenCL or CUDA program on a given context. A program is a collection of one or more device kernels which form a single compilation unit together. The device-code is passed as a string. Such a string can for example be generated, hard-coded, or read from file at run-time. If passed as an r-value (e.g. using `std::move`), the device-code string is moved instead of copied into the class' member variable.

* `Program(const Device &device, const Context &context, const std::string& binary)`:
As above, but now the program is constructed based on an already compiled IR or binary of the device kernels. This requires a context corresponding to the binary. This constructor for OpenCL is based on the `clCreateProgramWithBinary` function.

Public method(s):

* `BuildStatus Build(const Device &device, std::vector<std::string> &options)`:
This method invokes the OpenCL or CUDA compiler to build the program at run-time for a specific target device. Depending on the back-end, specific options can be passed to the compiler in the form of the `options` vector. It returns whether or not compilation errors were generated by the run-time compiler.

* `std::string GetBuildInfo(const Device &device) const`:
Retrieves all compiler warnings and errors generated by the build process.

* `std::string GetIR() const`:
Retrieves the intermediate representation (IR) of the compiled program. When using the CUDA back-end, this returns the PTX-code. For the OpenCL back-end, this returns either an IR (e.g. PTX) or a binary. This is different per OpenCL implementation.

CLCudaAPI::Queue
-------------

Constructor(s):

* `Queue(const Context &context, const Device &device)`:
Creates a new queue to enqueue kernel launches and device memory operations. This is analogous to an OpenCL command queue or a CUDA stream.

Public method(s):

* `void Finish(Event &event) const` and `void Finish() const`:
Completes all tasks in the queue. In the case of the CUDA back-end, the first form additionally synchronizes on the specified event.

* `Context GetContext() const`:
Retrieves the CUDA/OpenCL context associated with this queue.

* `Device GetDevice() const`:
Retrieves the CUDA/OpenCL device associated with this queue.


template \<typename T\> CLCudaAPI::BufferHost
-------------

Constructor(s):

* `BufferHost(const Context &, const size_t size)`:
Initializes a new linear 1D memory buffer on the host of type T. This buffer is allocated with a fixed number of elements given by `size`. Note that the buffer's elements are not initialized. In the case of the CUDA back-end, this host buffer is implemented as page-locked memory. The OpenCL back-end uses a regular `std::vector` container.

Public method(s):

* `size_t GetSize() const`:
Retrieves the allocated size in bytes.

* Several `std::vector` methods:
Adds some compatibility with `std::vector` by implementing the `size`, `begin`, `end`, `operator[]`, and `data` methods.


template \<typename T\> CLCudaAPI::Buffer
-------------

Constants(s):

* `enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned }`
Defines the different access types for the buffers. Writing to a read-only buffer will throw an error, as will reading from a write-only buffer. A buffer which is of type `kNotOwned` will not be automatically freed afterwards.

Constructor(s):

* `Buffer(const Context &context, const BufferAccess access, const size_t size)`:
Initializes a new linear 1D memory buffer on the device of type T. This buffer is allocated with a fixed number of elements given by `size`. Note that the buffer's elements are not initialized. The buffer can be read-only, write-only, read-write, or not-owned as specified by the `access` argument.

* `Buffer(const Context &context, const size_t size)`:
As above, but now defaults to read-write access.

* `template <typename Iterator> Buffer(const Context &context, const Queue &queue, Iterator start, Iterator end)`:
Creates a new buffer based on data in a linear C++ container (such as `std::vector`). The size is determined by the difference between the end and start iterators. This method both creates a new buffer and writes data to it. It synchronises the queue before returning.

Public method(s):

* `void ReadAsync(const Queue &queue, const size_t size, T* host) const` and
`void ReadAsync(const Queue &queue, const size_t size, std::vector<T> &host)` and
`void ReadAsync(const Queue &queue, const size_t size, BufferHost<T> &host)`:
Copies `size` elements from the current device buffer to the target host buffer. The host buffer has to be pre-allocated with a size of at least `size` elements. This method is a-synchronous: it can return before the copy operation is completed.

* `void Read(const Queue &queue, const size_t size, T* host) const` and
`void Read(const Queue &queue, const size_t size, std::vector<T> &host)` and
`void Read(const Queue &queue, const size_t size, BufferHost<T> &host)`:
As above, but now completes the operation before returning.

* `void WriteAsync(const Queue &queue, const size_t size, const T* host)` and
`void WriteAsync(const Queue &queue, const size_t size, const std::vector<T> &host)` and
`void WriteAsync(const Queue &queue, const size_t size, const BufferHost<T> &host)`:
Copies `size` elements from a host buffer to the current device buffer. The device buffer has to be pre-allocated with a size of at least `size` elements. This method is a-synchronous: it can return before the copy operation is completed.

* `void Write(const Queue &queue, const size_t size, const T* host)` and
`void Write(const Queue &queue, const size_t size, const std::vector<T> &host)` and
`void Write(const Queue &queue, const size_t size, const BufferHost<T> &host)`:
As above, but now completes the operation before returning.

* `void CopyToAsync(const Queue &queue, const size_t size, const Buffer<T> &destination) const`:
Copies `size` elements from the current device buffer to another device buffer given by `destination`. The destination buffer has to be pre-allocated with a size of at least `size` elements. This method is a-synchronous: it can return before the copy operation is completed.

* `void CopyTo(const Queue &queue, const size_t size, const Buffer<T> &destination) const`:
As above, but now completes the operation before returning.

* `size_t GetSize() const`:
Retrieves the allocated size in bytes.


CLCudaAPI::Kernel
-------------

Constructor(s):

* `Kernel(const Program &program, const std::string &name)`:
Retrieves a new kernel from a compiled program. The kernel name is given as the string `name`.

Public method(s):

* `template <typename T> void SetArgument(const size_t index, const T &value)`:
Method to set a kernel argument (l-value or r-value). The argument `index` specifies the position in the list of kernel arguments. The argument `value` can also be a `CLCudaAPI::Buffer`.

* `template <typename... Args> void SetArguments(Args&... args)`: As above, but now sets all arguments in one go, starting at index 0. This overwrites any previous arguments (if any). The parameter pack `args` takes any number of arguments of different types, including `CLCudaAPI::Buffer`.

* `unsigned long LocalMemUsage(const Device &device) const`:
Retrieves the amount of on-chip scratchpad memory (local memory in OpenCL, shared memory in CUDA) required by this specific kernel.

* `std::string GetFunctionName() const `:
Retrieves the name of the kernel (OpenCL only).

* `Launch(const Queue &queue, const std::vector<size_t> &global, const std::vector<size_t> &local, Event &event)`:
Launches a kernel onto the specified queue. This kernel launch is a-synchronous: this method can return before the device kernel is completed. The total number of threads launched is equal to the `global` vector; the number of threads per OpenCL work-group or CUDA thread-block is given by the `local` vector. The elapsed time is recorded into the `event` argument.

* `Launch(const Queue &queue, const std::vector<size_t> &global, const std::vector<size_t> &local, Event &event, std::vector<Event>& waitForEvents)`: As above, but now this kernel is only launched after the other specified events have finished (OpenCL only). If `local` is empty, the kernel-size is determined automatically (OpenCL only).

