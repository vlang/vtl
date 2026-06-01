# VSL ↔ VTL CUDA alignment

VTL `-d cuda` builds require a recent **vsl** with public CUDA memcpy kind constants.

## Required VSL changes

In `vsl/cuda/_cfun.c.v`:

- `pub const cuda_memcpy_host_to_device`, `cuda_memcpy_device_to_host`, `cuda_memcpy_device_to_device`
- `#flag` defines all three kinds for the C preprocessor (`-Dcuda_memcpy_device_to_device=4`)

In `vsl/cuda/compute/optimizer_gpu.v`:

- Use `cuda.cuda_memcpy_device_to_device` (not `C.cuda_memcpy_device_to_device`)

## Verify locally

```bash
cd ~/.vmodules/vtl
VTL_USE_CUDA=1 VJOBS=1 v -d cuda test nn/layers/linear_cuda_test.v
VTL_USE_CUDA=1 VJOBS=1 v -d cuda run examples/nn_cifar10_cuda/main.v
```

## Pin

After VSL merges `fix/pub-cuda-memcpy-kinds`, bump the vsl revision in the vtl module manifest if applicable.
