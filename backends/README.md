# All Backends in VTL Engine

VTL Engine supports the following backends:
   - Cuda
   - OpenCL
   - OpenMP
   - Vulkan

## Cuda
 First install cuda toolkit: https://developer.nvidia.com/cuda-downloads
  ### usage:
  ```vlang
        import vtl
        vtl_tensor := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
        vtl_tensor.cuda()
        println('My device: ${vtl_tensor.device}')

  ```

## OpenCL
  First install opencl toolkit: https://www.khronos.org/opencl/
  ### usage:
  ```vlang
        import vtl
        vtl_tensor := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
        vtl_tensor.opencl()
        println('My device: ${vtl_tensor.device}')

  ```

## OpenMP
  ### usage:
  ```vlang
        import vtl
        vtl_tensor := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
        vtl_tensor.openmp()
        println('My device: ${vtl_tensor.device}')

  ```

## Vulkan
  ### usage:
  ```vlang
        import vtl
        vtl_tensor := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
        vtl_tensor.vulkan()
        println('My device: ${vtl_tensor.device}')

  ```
