module main

import vtl
import vtl.nn.layers

// Minimal Vulkan f32 Linear forward smoke (no autograd / Sequential — f32 models still compile-heavy).
//
//   VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
//
// Without VTL_USE_VULKAN=1, stays on CPU la.matmul path.

fn main() {
	use_vk := layers.vulkan_linear_enabled()
	println('nn_cifar10_vulkan: VTL_USE_VULKAN=${use_vk} (build with -d vulkan for GPU GEMM)')

	// CIFAR-shaped flattened batch: [4, 3072] @ [3072, 10] + bias
	m := 4
	k := 3 * 32 * 32
	n := 10
	x := vtl.ones[f32]([m, k])
	w := vtl.zeros[f32]([n, k])
	b := vtl.zeros[f32]([1, n])
	out := layers.linear_forward_f32(x, w, b) or { panic(err) }
	assert out.shape == [m, n]

	println('Linear forward OK shape=${out.shape} ✅')
}
