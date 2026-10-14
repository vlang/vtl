# VTL — ML Roadmap & Launch Tracking

Maintainer planning: **https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL; may require project access)

Detailed roadmap: [ROADMAP.md](../ROADMAP.md)

GPU memory: [DEVICE_MEMORY.md](DEVICE_MEMORY.md)

## Completed issues

| Issue | Topic |
|-------|--------|
| [#87](https://github.com/vlang/vtl/issues/87) | Serialization + CIFAR checkpoints |
| [#88](https://github.com/vlang/vtl/issues/88) | vs NumPy/PyTorch benchmarks + PR comments |
| [#89](https://github.com/vlang/vtl/issues/89)–[#91](https://github.com/vlang/vtl/issues/91) | CUDA Linear/Conv2D + `DeviceSession` (Phase 1) |
| [#101](https://github.com/vlang/vtl/issues/101)/[#104](https://github.com/vlang/vtl/pull/104) | GPU activation chain (Phase 2) |
| [#105](https://github.com/vlang/vtl/pull/105) | Linear CUDA backward (`VTL_CUDA_BACKWARD=1`, Phase 3) |
| [#107](https://github.com/vlang/vtl/issues/107) | Conv2D CUDA backward (cuDNN, same eligibility as forward) |
| [#111](https://github.com/vlang/vtl/pull/111)/[#114](https://github.com/vlang/vtl/pull/114) | Adam on GPU + persistent `DeviceSession` slots (#106) |
| [#110](https://github.com/vlang/vtl/issues/110) | Vulkan f32 Linear in `Sequential` forward (`VTL_USE_VULKAN=1`, `-d vulkan`) |
| [#116](https://github.com/vlang/vtl/issues/116) | f32 autograd: `Sequential` + MSE forward/backprop compile |
| — | f32 tiny training: `nn_cifar10_f32_tiny_synth` + `f32_training_smoke_test` |
| — | CUDA training smoke: `nn_cifar10_cuda` + `nn/cuda_training_smoke_test` |
| — | f32 Vulkan training: `nn_cifar10_vulkan` + `f32_vulkan_training_smoke_d_vulkan_test` |
| — | Vulkan Conv2D f32 forward/backward (same-padding, im2col+GEMM) |
| — | Vulkan ReLU/Sigmoid/Softplus/SELU/HardSwish f32; experimental per-layer dispatch with CPU fallback |
| — | Vulkan Adam f32 fused shader (`VTL_USE_VULKAN=1`, VSL `adam_step`) |
| — | Conv2D autograd: register weight/bias parents (`conv2d_autograd_smoke_test`) |
| [#86](https://github.com/vlang/vtl/issues/86) | `DataLoader` |
| [#148](https://github.com/vlang/vtl/issues/148) | Contribution standards and test conventions |
| [#152](https://github.com/vlang/vtl/issues/152) | NAdam and RAdam optimizers |
| [#153](https://github.com/vlang/vtl/issues/153) | L1/MAE, Hinge, and Focal losses |
| [#154](https://github.com/vlang/vtl/issues/154) | GRU layer |
| [#155](https://github.com/vlang/vtl/issues/155) | Conv1D layer |
| [#157](https://github.com/vlang/vtl/issues/157) | Expanded Windows test coverage |
| [#158](https://github.com/vlang/vtl/issues/158) | Gradient-check utility |
| [#159](https://github.com/vlang/vtl/issues/159) | Clamp/clip autograd gate |
| [#160](https://github.com/vlang/vtl/issues/160) | Reshape/transpose/concat backward gates |
| [#162](https://github.com/vlang/vtl/issues/162) | Runnable OpenCL/VCL CI example |
| [#163](https://github.com/vlang/vtl/issues/163) | Transformer/attention end-to-end example |
| — | `from_array` clones shape (fixes [#41](https://github.com/vlang/vtl/issues/41) aliasing) |

**VSL (downstream):** [#280](https://github.com/vlang/vsl/issues/280)–[#285](https://github.com/vlang/vsl/issues/285), [#304](https://github.com/vlang/vsl/pull/304) conv2d backward GEMM, [#305](https://github.com/vlang/vsl/pull/305) Adam shaders.

## Closed issue status corrections

Issue [#62](https://github.com/vlang/vtl/issues/62) is closed on GitHub, but
its full VTL OpenCL backend checklist is not complete in the current tree. VTL
can transfer tensors through VCL storage and the [#162 OpenCL/VCL smoke
example](../examples/vtl_opencl_vcl_support/README.md) checks a round-trip and
`add_scalar`. `VclTensor` still has no VTL GEMM/GEMV, elementwise, reduction,
softmax, LayerNorm, Conv2D, or neural-network dispatch path. Track this as
unfinished backend work; issue closure is not implementation evidence. The
itemized status is in the [main roadmap](../ROADMAP.md#phase-5--opencl-backend).

## Current open issues (checked 2026-10-14)

| Issue | Topic |
|-------|-------|
| [#63](https://github.com/vlang/vtl/issues/63) | Phase 6: ARM GPU Support — Android, iOS, Embedded |
| [#161](https://github.com/vlang/vtl/issues/161) | feat: CUDA backward for LSTM, Attention, BatchNorm, LayerNorm, Embedding, Pooling |
| [#717](https://github.com/vlang/vtl/issues/717) | feat: complete NumPy dtype promotion for tensor arithmetic |
| [#718](https://github.com/vlang/vtl/issues/718) | feat: unify dtype and device options for tensor creation |
| [#719](https://github.com/vlang/vtl/issues/719) | feat: define NumPy-compatible copy and view semantics |
| [#720](https://github.com/vlang/vtl/issues/720) | feat: complete NumPy bounds modes across indexing APIs |
| [#721](https://github.com/vlang/vtl/issues/721) | feat: complete NumPy reduction keepdims and accumulator dtype coverage |
| [#722](https://github.com/vlang/vtl/issues/722) | feat: expand complex tensor and linear algebra coverage |
| [#723](https://github.com/vlang/vtl/issues/723) | feat: complete NumPy random distributions and generator streams |
| [#724](https://github.com/vlang/vtl/issues/724) | perf: validate FFT dtype and strided-array parity |
| [#725](https://github.com/vlang/vtl/issues/725) | feat(io): add NumPy text array interoperability |
| [#726](https://github.com/vlang/vtl/issues/726) | feat(io): add memory-mapped NumPy array access |
| [#727](https://github.com/vlang/vtl/issues/727) | feat: connect masked arrays to autograd and neural networks |
| [#728](https://github.com/vlang/vtl/issues/728) | feat: define structured-array and table interoperability scope |
| [#729](https://github.com/vlang/vtl/issues/729) | perf: publish a reproducible NumPy backend benchmark matrix |
| [#730](https://github.com/vlang/vtl/issues/730) | docs: add end-to-end NumPy and classical ML migration examples |
| [#731](https://github.com/vlang/vtl/issues/731) | feat: complete PyTorch tensor operator semantics audit |
| [#732](https://github.com/vlang/vtl/issues/732) | feat: define a unified PyTorch-style dtype and device model |
| [#733](https://github.com/vlang/vtl/issues/733) | feat: add higher-order autograd and graph controls |
| [#734](https://github.com/vlang/vtl/issues/734) | feat(nn): add stacked and bidirectional recurrent APIs |
| [#735](https://github.com/vlang/vtl/issues/735) | feat(nn): add parameter and buffer registration for custom modules |
| [#736](https://github.com/vlang/vtl/issues/736) | feat(optim): integrate schedulers and complete optimizer parity |
| [#737](https://github.com/vlang/vtl/issues/737) | feat(data): add composable transforms and reproducible streaming datasets |
| [#738](https://github.com/vlang/vtl/issues/738) | feat(data): add worker loading, prefetch and pinned-memory batches |
| [#739](https://github.com/vlang/vtl/issues/739) | feat: add safe checkpoint migration and device remapping |
| [#740](https://github.com/vlang/vtl/issues/740) | feat: add graph capture and fused tensor execution |
| [#741](https://github.com/vlang/vtl/issues/741) | perf: add tensor memory planning and profiling tools |
| [#742](https://github.com/vlang/vtl/issues/742) | feat: add distributed training and mixed-precision primitives |
| [#743](https://github.com/vlang/vtl/issues/743) | feat: expand VTL model interoperability and deployment |
| [#746](https://github.com/vlang/vtl/issues/746) | feat: add sparse tensor formats and operations |
| [#747](https://github.com/vlang/vtl/issues/747) | feat: define non-numeric NumPy dtype support |
| [#748](https://github.com/vlang/vtl/issues/748) | feat: add explicit array interoperability APIs |
| [#750](https://github.com/vlang/vtl/issues/750) | feat(opencl): expand tensor compute and neural-network dispatch |
| [#758](https://github.com/vlang/vtl/issues/758) | feat(cuda): keep optimizer state resident across training steps |
| [#770](https://github.com/vlang/vtl/issues/770) | docs: add PyTorch migration and training tutorials |
| [#771](https://github.com/vlang/vtl/issues/771) | perf: add matched VTL and PyTorch benchmarks |
| [#773](https://github.com/vlang/vtl/issues/773) | feat: add NumPy polynomial families and fitting |
| [#777](https://github.com/vlang/vtl/issues/777) | feat(math): complete NumPy ufunc family coverage |
| [#779](https://github.com/vlang/vtl/issues/779) | feat(io): add optional HDF5 tensor interoperability |
| [#783](https://github.com/vlang/vtl/issues/783) | feat(math): define NumPy ufunc execution methods |
| [#784](https://github.com/vlang/vtl/issues/784) | feat(autograd): add functional tensor transforms |
| [#792](https://github.com/vlang/vtl/issues/792) | feat(quantization): add INT8 tensor and inference support |
| [#797](https://github.com/vlang/vtl/issues/797) | feat(vulkan): retain activations across sequential layers |

Issues #3 and #40 were closed on GitHub on 2026-10-13. The #3 resolution
uses explicit SIMD and validated contiguous kernels rather than a V-level
`restrict` contract; V 0.5.2 does not emit a non-aliasing qualifier, and
`@[direct_array_access]` removes bounds checks without establishing alias
guarantees. The #40 resolution reports single-traversal backward coverage for
every autograd gate family; broader operation-chain fusion remains in the
performance backlog below. Issue #52's Burn comparison is documented in
[BURN_PARITY.md](./BURN_PARITY.md) and is closed. Issue #41's Windows heap
corruption was fixed in main (PR [#409](https://github.com/vlang/vtl/pull/409)).
## Additional post-beta tracking

| Priority | Issue | Topic |
|----------|-------|--------|
| P2 | [#797](https://github.com/vlang/vtl/issues/797) | Vulkan: preserve activations across sequential layers |

## Performance engineering backlog

The historical [#64 performance engineering proposal](https://github.com/vlang/vtl/issues/64)
is closed, but its acceptance checklist is not complete in the current tree.
VTL has specific optimized paths, including fused Vulkan Adam, while the
broader work remains: general fused operation chains, FP16/BF16 and automatic
mixed precision, reusable GPU memory pools, asynchronous Vulkan execution,
backend profiling, and checkpointing. Treat these as future work until each
has an implementation, correctness coverage, and benchmark evidence; issue
closure alone is not completion evidence.

The beta contract is tensors, autograd, high-level layers/losses/optimizers,
datasets, and f32 CPU training. CUDA and Vulkan remain opt-in experimental
backend paths during beta. Generic f64 training is tracked as post-beta until
the autograd `Gate[T]` interface compiles without mixed `Payload[f32]` and
`Payload[f64]` specialization.

## Local development

**Lightweight workflow:** [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md)

```bash
v up
cd ~/.vmodules
v test ./vtl/nn ./vtl/datasets
v run ./vtl/examples/nn_cifar10_tiny_synth/main.v
v run ./vtl/examples/nn_cifar10_f32_tiny_synth/main.v
# Vulkan f32 full stack (use -prod for GPU)
# VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
# VTL_USE_VULKAN=1 VTL_TEST_VULKAN=1 VJOBS=1 v -prod -d vulkan test vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v
```

## CI

- **Default PR:** `bin/test` (scoped modules + `nn_cifar10_tiny_synth`)
- **f32 beta smoke:** `nn_cifar10_f32_tiny_synth` and f32 autograd/training tests
- **`full-ml` label:** `ci-full-ml.yml` runs `bin/test --full`
- **CUDA / Vulkan:** local or future labeled workflows

## Project board sync

From repo root (requires `gh` scopes `read:project`, `project`):

```bash
./.github/scripts/sync-ml-project-8.sh
```
