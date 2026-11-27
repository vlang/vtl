# Resident Vulkan activation inference

This example runs a two-layer `Sequential` ReLU/Sigmoid chain through the
opt-in resident f32 inference path. Adjacent supported activations share one
Vulkan tensor and only convert back to a CPU tensor at the end. The example
uses the regular Sequential implementation if Vulkan is not enabled. Vulkan
device, upload, dispatch, and readback errors are reported rather than hidden.

```sh
VTL_USE_VULKAN=1 v -prod -d vulkan run examples/nn_vulkan_resident_inference/main.v
```

The API is inference-only: call `eval()`, enable `ctx.no_grad`, and pass an
untracked input variable. This initial slice supports only ReLU and Sigmoid;
unsupported layers return to ordinary Sequential execution. Vulkan kernels
still synchronously wait for each dispatch, so this reduces host transfers but
does not yet remove per-layer queue synchronization.
