# Tensor + OpenCL + VCL support

This example shows how to use the OpenCL backend with VCL support.

## Running the CI smoke example

Run from `~/.vmodules` so V resolves both local modules:

```sh
v -d vcl run vtl/examples/vtl_opencl_vcl_support/main.v
```

The example checks for an OpenCL device and exits successfully with a `SKIP`
message when none is available. With a device, it round-trips a VTL tensor
through VCL and checks a scalar addition kernel. CI uses PoCL to provide a CPU
OpenCL device.

## Prerequisites

Read the [VCL docs](https://vlang.github.io/vsl/vcl.html) and the OpenCL backend guide.

## Running the example

The original manual example is also available:

```sh
v -d vcl run vtl/examples/vtl_opencl_vcl_support/main_not_ci.v
```
