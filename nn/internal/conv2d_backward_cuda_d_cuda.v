module internal

import vsl.cuda
import vsl.cuda.compute
import vtl
import vtl.autograd_cuda

// conv2d_backward_f64 exposes this operation as part of the public API.
pub fn conv2d_backward_f64(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f64] {
	if !autograd_cuda.cuda_backward_enabled() || !conv2d_cuda_eligible(kernel_size, config) {
		return conv2d_backward_cpu_f64(grad_out, input, weight, bias, kernel_size, config)
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
	bwd := compute.conv2d_backward_cuda(dev, grad_out.to_array(), input.to_array(),
		weight.to_array(), batch, in_h, in_w, in_ch, out_ch, k_h, k_w, stride_h, stride_w)!

	d_input := vtl.from_array(bwd.d_input, input.shape)!
	d_weight := vtl.from_array(bwd.d_weight, weight.shape)!
	mut d_bias := vtl.zeros[f64]([1, out_ch])
	out_h := grad_out.shape[2]
	out_w := grad_out.shape[3]
	for oc in 0 .. out_ch {
		mut sum := 0.0
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
