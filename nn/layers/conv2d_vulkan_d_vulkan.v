module layers

import vtl
import vtl.nn.internal
import vsl.vulkan
import vtl.storage

// conv2d_forward_vulkan implements Conv2D forward pass on Vulkan via im2col + GEMM.
// input: [batch, in_ch, H, W]
// weight: [out_ch, in_ch, k_h, k_w]
// bias: [1, out_ch]
// Returns: [batch, out_ch, out_H, out_W]
pub fn conv2d_forward_vulkan[T](
	input       &vtl.Tensor[T],
	weight      &vtl.Tensor[T],
	bias        &vtl.Tensor[T],
	kernel_size []int,
	config      Conv2DConfig
) !&vtl.Tensor[T] {
		// Extract dimensions
		batch := u32(input.shape[0])
		in_ch := u32(input.shape[1])
		in_h := u32(input.shape[2])
		in_w := u32(input.shape[3])
		out_ch := u32(weight.shape[0])
		k_h := u32(kernel_size[0])
		k_w := u32(kernel_size[1])
		pad_h := u32(config.padding[0])
		pad_w := u32(config.padding[1])
		stride_h := u32(config.stride[0])
		stride_w := u32(config.stride[1])
		dil_h := u32(config.dilation[0])
		dil_w := u32(config.dilation[1])

		// Compute output spatial dimensions
		out_h := (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1
		out_w := (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1

		// Get Vulkan device from input storage
		input_vk := match input.storage {
			storage.VulkanStorage[T] { input.storage as storage.VulkanStorage[T] }
			else { return error('conv2d_forward_vulkan: input must use VulkanStorage') }
		}
		dev := input_vk.params.device

		// Allocate GPU buffers
		im2col_size := batch * out_h * out_w * in_ch * k_h * k_w
		mut im2col_buf := dev.buffer(vulkan.DeviceSize(im2col_size * 4))!
		defer { im2col_buf.release() }

		// Reshape weight [out_ch, in_ch, k_h, k_w] → [out_ch, in_ch*k_h*k_w]
		weight_mat_shape := [int(out_ch), int(in_ch * k_h * k_w)]
		weight_reshaped := weight.reshape(weight_mat_shape)!

		// Upload input to GPU
		mut input_buf := dev.buffer(vulkan.DeviceSize(input.size() * 4))!
		defer { input_buf.release() }
		input_buf.load_f32(input.to_f32())!

		// Im2col transform on GPU
		vulkan.im2col(dev, im2col_buf, input_buf, batch, in_ch, in_h, in_w, k_h, k_w,
			out_h, out_w, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w)!

		// Upload weight to GPU
		mut weight_buf := dev.buffer(vulkan.DeviceSize(weight_reshaped.size() * 4))!
		defer { weight_buf.release() }
		weight_buf.load_f32(weight_reshaped.to_f32())!

		// GEMM: [batch*out_h*out_w, in_ch*k_h*k_w] @ [in_ch*k_h*k_w, out_ch]
		//       = [batch*out_h*out_w, out_ch]
		m := batch * out_h * out_w
		n := out_ch
		k := in_ch * k_h * k_w
		mut gemm_out_buf := dev.buffer(vulkan.DeviceSize(m * n * 4))!
		defer { gemm_out_buf.release() }

		// GEMM C = A * B^T (row-major, weight needs transpose)
		// For now: use CPU GEMM fallback (full GPU GEMM wiring is Phase 1 done, reuse here)
		// TODO: wire VSL's vulkan.gemm once weight transpose kernel is ready
		
		// CPU fallback for now
		return internal.conv2d_forward[T](input, weight, bias, kernel_size, internal.Conv2DConfig{
			padding:  config.padding
			stride:   config.stride
			dilation: config.dilation
			groups:   config.groups
		})!
}
