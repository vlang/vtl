# Tutorial: GPU Backends (Vulkan + OpenCL/VCL)

This tutorial explains how to run VTL workloads on GPU backends and benchmark them against CPU.

## Current backend status

VTL uses the unified backend-agnostic entrypoint from VSL:

- `vsl.compute` for operation dispatch
- backend implementations under `vsl/vcl/compute` and `vsl/vulkan/compute`

- **Vulkan backend (`-d vulkan`)**
  - Autograd gates and training-related paths are available for selected operations.
  - Tensor and NN integrations are available for Vulkan-enabled builds.
- **OpenCL backend via VCL (`-d vcl`)**
  - `VclTensor` conversion and compute ops are available.
  - LA integration includes `la.matmul_vcl` and VCL-backed tensor operations.

## Build flags

Use compile-time flags to activate GPU code paths:

```sh
# CPU only
v run your_program.v

# OpenCL/VCL backend
v -d vcl run your_program.v

# Vulkan backend
v -d vulkan run your_program.v

# Both enabled
v -d vulkan -d vcl run your_program.v
```

Compile-time flags define which backends are available in the binary.
Runtime policy selects which backend to use at execution time.

## Runtime backend selection

You can choose backend at runtime without changing model code:

```sh
# Auto selection (default)
v -d vulkan -d vcl run your_program.v

# Force Vulkan
VTL_BACKEND=vulkan v -d vulkan -d vcl run your_program.v

# Force OpenCL/VCL
VTL_BACKEND=vcl v -d vulkan -d vcl run your_program.v

# Force CPU
VTL_BACKEND=cpu v -d vulkan -d vcl run your_program.v

# Strict mode: fail if preferred backend is unavailable
VTL_BACKEND=vulkan VTL_BACKEND_STRICT=1 v -d vulkan -d vcl run your_program.v
```

Supported values for `VTL_BACKEND`: `auto`, `cpu`, `vulkan`, `vcl`.

Current runtime-dispatched NN/LA coverage:

- `la.matmul` through `Linear` layer
- `ReLU`, `Sigmoid`, `Tanh` layers
- `Conv2D` layer (tries Vulkan path first when selected, then CPU fallback when not strict)
- `MaxPool2D`, `AveragePool2D`, `GlobalAveragePool2D` layers (Vulkan try + CPU fallback)
- `Softmax` layer for 1-D tensors (Vulkan try + CPU fallback)
- `BatchNorm1D` inference path (Vulkan normalize + CPU affine/fallback)

You can also configure backend policy directly on a model context:

```v ignore
import vsl.compute
import vtl.autograd
import vtl.nn.models

mut ctx := autograd.ctx[f64]()
mut model := models.sequential_from_ctx[f64](ctx)
model.set_backend(compute.Backend.vulkan)
model.set_backend_strict(false)
```

## OpenCL/VCL workflow

```v ignore
import vtl
import vtl.storage

// CPU tensor
x := vtl.from_2d[f64]([[1.0, 2.0], [3.0, 4.0]])!

// Move to VCL storage (device auto-selected when device is nil)
vx := x.vcl(storage.VclStorageParams{})!

// Run VCL-backed tensor ops
vy := vx.relu()!
vz := vy.add_scalar(1.0)!

// Bring back to CPU
cpu := vz.cpu()!
println(cpu)
```

## Vulkan workflow

```v ignore
import vtl
import vtl.storage

x := vtl.from_2d[f32]([[1.0, 2.0], [3.0, 4.0]])!
params := storage.VulkanStorageParams{}

vx := x.vulkan(params)!
vy := vx.relu()!
out := vy.cpu()!
println(out)
```

## Linear algebra on OpenCL (via unified compute dispatch)

```v ignore
import vtl
import vtl.la

// A[2x2] * B[2x2]
a := vtl.from_2d[f64]([[1, 2], [3, 4]])!
b := vtl.from_2d[f64]([[5, 6], [7, 8]])!

c := la.matmul_vcl[f64](a, b)!
println(c)
```

## Backend benchmark module

VTL includes a benchmark module to compare CPU, Vulkan, and VCL:

```sh
# CPU only
v run benchmark/main.v

# CPU + OpenCL/VCL
v -d vcl run benchmark/main.v

# CPU + Vulkan
v -d vulkan run benchmark/main.v

# CPU + Vulkan + VCL
v -d vulkan -d vcl run benchmark/main.v
```

See [`benchmark/README.md`](../benchmark/README.md) for details.

## MNIST and GPU notes

The `examples/nn_mnist` example demonstrates a full training pipeline.
GPU backends can be enabled through flags, but acceleration depends on
the exact ops used by the model and current backend coverage.

For practical guidance, see:

- [`examples/nn_mnist/README.md`](../examples/nn_mnist/README.md)
- [`examples/vtl_opencl_vcl_support/README.md`](../examples/vtl_opencl_vcl_support/README.md)

## Troubleshooting

- If `-d vcl` fails, verify OpenCL runtime/ICD installation and device availability.
- If `-d vulkan` fails, verify Vulkan loader + GPU driver.
- Keep backend and compiler versions aligned with current VSL/VTL main branches.
