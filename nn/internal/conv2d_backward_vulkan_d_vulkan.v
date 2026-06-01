module internal

import os
import vtl
import vsl.vulkan.compute

// conv2d_backward_vulkan_f32: GPU d_weight via GEMM; d_input/d_bias on CPU (VSL reference).
pub fn conv2d_backward_vulkan_f32(grad_out &vtl.Tensor[f32], input &vtl.Tensor[f32], weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f32] {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return error('conv2d_backward_vulkan_f32: set VTL_USE_VULKAN=1')
	}
	if !conv2d_vulkan_backward_eligible(kernel_size, config) {
		return error('conv2d_backward_vulkan_f32: config not supported')
	}

	batch := input.shape[0]
	in_ch := input.shape[1]
	in_h := input.shape[2]
	in_w := input.shape[3]
	out_ch := weight.shape[0]
	k_h := kernel_size[0]
	k_w := kernel_size[1]
	stride_h := config.stride[0]
	stride_w := config.stride[1]
	pad_h := config.padding[0]
	pad_w := config.padding[1]

	dev := conv2d_vulkan_device()!
	grad_flat := f32_flat_to_f64(grad_out.to_array())
	in_flat := f32_flat_to_f64(input.to_array())
	w_flat := f32_flat_to_f64(weight.to_array())

	bwd := compute.conv2d_backward_vulkan(dev, grad_flat, in_flat, w_flat, batch, in_h, in_w, in_ch,
		out_ch, k_h, k_w, stride_h, stride_w, pad_h, pad_w)!

	mut d_weight_f32 := []f32{len: bwd.d_weight.len}
	for i, v in bwd.d_weight {
		d_weight_f32[i] = f32(v)
	}
	d_input := vtl.from_array(f64_to_f32(bwd.d_input), input.shape)!
	d_weight := vtl.from_array(d_weight_f32, weight.shape)!

	mut d_bias := vtl.zeros[f32]([1, out_ch])
	out_h := grad_out.shape[2]
	out_w := grad_out.shape[3]
	for oc in 0 .. out_ch {
		mut sum := f32(0)
		for b in 0 .. batch {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					sum += grad_out.get([b, oc, oh, ow])
				}
			}
		}
		d_bias.set([0, oc], sum)
	}
	_ = bias
	return [d_input, d_weight, d_bias]
}

fn f64_to_f32(arr []f64) []f32 {
	mut out := []f32{len: arr.len}
	for i, v in arr {
		out[i] = f32(v)
	}
	return out
}
