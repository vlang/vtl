module internal

import os
import vtl
import vtl.storage
import vsl.vulkan
import vsl.vulkan.compute

// conv2d_vulkan_eligible: same-padding as CUDA/cuDNN path (groups=1, dilation=1) for forward.
pub fn conv2d_vulkan_eligible(kernel_size []int, config Conv2DConfig) bool {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return false
	}
	if config.groups != 1 {
		return false
	}
	if config.dilation != [1, 1] {
		return false
	}
	k_h := kernel_size[0]
	k_w := kernel_size[1]
	expected_pad_h := (k_h - 1) / 2
	expected_pad_w := (k_w - 1) / 2
	return config.padding == [expected_pad_h, expected_pad_w]
}

// conv2d_vulkan_backward_eligible: GPU d_weight GEMM validated for padding=0 only.
pub fn conv2d_vulkan_backward_eligible(kernel_size []int, config Conv2DConfig) bool {
	if !conv2d_vulkan_eligible(kernel_size, config) {
		return false
	}
	return config.padding == [0, 0]
}

fn f32_flat_to_f64(arr []f32) []f64 {
	mut out := []f64{len: arr.len}
	for i, v in arr {
		out[i] = f64(v)
	}
	return out
}

fn conv2d_vulkan_device() !&vulkan.Device {
	mut probe := vtl.zeros[f32]([1])
	vk := probe.vulkan(storage.VulkanParams{})!
	defer { vk.release() }
	return vk.data.data.device
}

// conv2d_forward_vulkan_f32 runs VSL im2col+GEMM; returns CPU tensor for autograd.
pub fn conv2d_forward_vulkan_f32(input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f32] {
	if !conv2d_vulkan_eligible(kernel_size, config) {
		return error('conv2d_forward_vulkan_f32: config not supported by Vulkan path')
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

	dev := conv2d_vulkan_device()!
	in_flat := f32_flat_to_f64(input.to_array())
	w_flat := f32_flat_to_f64(weight.to_array())

	pad_h := config.padding[0]
	pad_w := config.padding[1]
	out_flat := compute.conv2d_vulkan(dev, in_flat, w_flat, batch, in_h, in_w, in_ch, out_ch, k_h,
		k_w, stride_h, stride_w, pad_h, pad_w)!

	out_h := (in_h + 2 * pad_h - k_h) / stride_h + 1
	out_w := (in_w + 2 * pad_w - k_w) / stride_w + 1

	mut out_f32 := []f32{len: out_flat.len}
	for i, v in out_flat {
		out_f32[i] = f32(v)
	}
	for b in 0 .. batch {
		for oc in 0 .. out_ch {
			bv := bias.get([0, oc])
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					idx := ((b * out_ch + oc) * out_h + oh) * out_w + ow
					out_f32[idx] += bv
				}
			}
		}
	}

	return vtl.from_array(out_f32, [batch, out_ch, out_h, out_w])!
}
