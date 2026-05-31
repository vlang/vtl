module internal

import os
import vtl
import vsl.cuda
import vsl.cuda.compute

// conv2d_cuda_eligible reports whether VSL cuDNN conv2d matches this VTL config.
// cuDNN path uses padding (k-1)/2 and groups=1, dilation=1.
pub fn conv2d_cuda_eligible(kernel_size []int, config Conv2DConfig) bool {
	if os.getenv('VTL_USE_CUDA') != '1' {
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

// conv2d_forward_cuda_f64 runs forward conv on GPU; returns CPU tensor for autograd.
pub fn conv2d_forward_cuda_f64(input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f64] {
	if !conv2d_cuda_eligible(kernel_size, config) {
		return error('conv2d_forward_cuda_f64: config not supported by cuDNN path')
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

	dev := cuda.get_default_device()!
	in_flat := input.to_array()
	w_flat := weight.to_array()

	out_flat := compute.conv2d_cuda(dev, in_flat, w_flat, batch, in_h, in_w, in_ch, out_ch, k_h,
		k_w, stride_h, stride_w)!

	out_h := (in_h + 2 * config.padding[0] - k_h) / stride_h + 1
	out_w := (in_w + 2 * config.padding[1] - k_w) / stride_w + 1

	mut out_row := out_flat.clone()
	for b in 0 .. batch {
		for oc in 0 .. out_ch {
			bv := bias.get([0, oc])
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					idx := ((b * out_ch + oc) * out_h + oh) * out_w + ow
					out_row[idx] += bv
				}
			}
		}
	}

	return vtl.from_array(out_row, [batch, out_ch, out_h, out_w])!
}
