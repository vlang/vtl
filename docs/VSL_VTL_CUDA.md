# VSL ↔ VTL CUDA alignment

VTL `-d cuda` builds require a recent **vsl** with public CUDA memcpy kind constants.

## Required VSL changes

In `vsl/cuda/_cfun.c.v`:

- `pub const cuda_memcpy_host_to_device`
- `pub const cuda_memcpy_device_to_host`
- `pub const cuda_memcpy_device_to_device`
- `#flag` defines all three kinds for the C preprocessor (`-Dcuda_memcpy_device_to_device=4`)

In `vsl/cuda/compute/optimizer_gpu.v`:

- Use `cuda.cuda_memcpy_device_to_device` (not `C.cuda_memcpy_device_to_device`)

## Verify locally

```bash
cd ~/.vmodules/vtl
VTL_USE_CUDA=1 VJOBS=1 v -d cuda test nn/layers/linear_cuda_test.v
VTL_USE_CUDA=1 VJOBS=1 v -d cuda run examples/nn_cifar10_cuda/main.v
```

## Status

The public memcpy kind constants are available in VSL main. Keep this note as a
compatibility checklist for users pinning older VSL revisions.

## GPU backward status (ADR-002, phase 1)

Opt-in with `-d cuda`, `VTL_USE_CUDA=1` and `VTL_CUDA_BACKWARD=1`. The CPU path is the
default and the fallback.

| Op | GPU path | Limits | Fallback |
|----|----------|--------|----------|
| BatchNorm2D backward (training) | `cudnnBatchNormalizationBackwardEx`, spatial mode | epsilon >= 1e-5 | evaluation mode and smaller epsilon use CPU |
| MaxPool2D backward | `cudnnPoolingBackward` | padding 0; no equal maxima inside a window | padded or tied windows use CPU |
| LayerNorm backward | not implemented | - | CPU |

The cuDNN declarations live in `nn/internal/cudnn_norm_pool_d_cuda.v`. vsl binds only
forward and convolution APIs. The vsl constant `cuda.cudnn_tensor_nchw` is 1, which is
`CUDNN_TENSOR_NHWC` in cuDNN, so VTL uses 0 (`CUDNN_TENSOR_NCHW`) for NCHW descriptors.

LayerNorm is not on the GPU. The legacy `cudnnNormalizationBackward` has no LAYER mode, and
its scale and bias are per non-reduced dimension, so per-feature LayerNorm affine parameters
do not fit. The graph API (`CUDNN_LAYER_NORM` with `CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR`)
is the candidate for a follow-up.

Verify with `v -d cuda test` on each `*_cuda_test.v` file, with `VTL_TEST_CUDA=1`.
