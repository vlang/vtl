module layers

// Stub for non-Vulkan builds — conv2d_forward_vulkan falls through to CPU.
import vtl
import vtl.nn.internal

pub fn conv2d_forward_vulkan[T](input &vtl.Tensor[T],
	weight &vtl.Tensor[T],
	bias &vtl.Tensor[T],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[T] {
	cfg := internal.Conv2DConfig{
		padding:  config.padding
		stride:   config.stride
		dilation: config.dilation
		groups:   config.groups
	}
	return internal.conv2d_forward[T](input, weight, bias, kernel_size, cfg)
}
