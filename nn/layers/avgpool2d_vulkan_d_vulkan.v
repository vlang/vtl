module layers

import vtl
import vsl.vulkan

// AvgPool2DLayerVulkan implements 2D average pooling on GPU
pub struct AvgPool2DLayerVulkan[T] {
pub mut:
	kernel_size [2]int // [height, width]
	stride      [2]int // [height, width]
	padding     [2]int // [height, width]
	device      &vulkan.Device = unsafe { nil }
}

// avgpool2d_forward_vulkan performs 2D average pooling on GPU.
// input: [batch, channels, height, width]
// output: [batch, channels, out_h, out_w]
pub fn avgpool2d_forward_vulkan[T](input &vtl.Tensor[T], kernel_size [2]int, stride [2]int, padding [2]int, dev &vulkan.Device) !&vtl.Tensor[T] {
	// Extract input dimensions (NCHW layout)
	batch := u32(input.shape[0])
	channels := u32(input.shape[1])
	in_h := u32(input.shape[2])
	in_w := u32(input.shape[3])

	k_h := u32(kernel_size[0])
	k_w := u32(kernel_size[1])
	stride_h := u32(stride[0])
	stride_w := u32(stride[1])
	pad_h := u32(padding[0])
	pad_w := u32(padding[1])

	// Calculate output dimensions
	out_h := u32((int(in_h) + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
	out_w := u32((int(in_w) + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)

	// Allocate GPU buffers
	input_size := batch * channels * in_h * in_w * 4 // f32 = 4 bytes
	output_size := batch * channels * out_h * out_w * 4

	mut input_buf := dev.buffer(vulkan.DeviceSize(input_size))!
	defer { input_buf.release() }
	mut output_buf := dev.buffer(vulkan.DeviceSize(output_size))!
	defer { output_buf.release() }

	// Upload input data
	mut input_bytes := []u8{len: int(input_size)}
	for i := 0; i < int(batch * channels * in_h * in_w); i++ {
		val := f32(input.get([i / int(channels * in_h * in_w), (i / int(in_h * in_w)) % int(channels),
			(i / int(in_w)) % int(in_h), i % int(in_w)]))
		unsafe {
			*(&f32(&input_bytes[i * 4])) = val
		}
	}
	input_buf.load(input_bytes)!

	// Run avgpool2d kernel
	vulkan.avgpool2d(dev, output_buf, input_buf, batch, channels, in_h, in_w, k_h, k_w, out_h,
		out_w, pad_h, pad_w, stride_h, stride_w)!

	// Download output
	mut output_bytes := []u8{len: int(output_size)}
	output_bytes = output_buf.store(mut output_bytes)!

	// Convert to tensor
	mut output_data := []T{len: int(batch * channels * out_h * out_w)}
	for i in 0 .. int(batch * channels * out_h * out_w) {
		unsafe {
			val := *(&f32(&output_bytes[i * 4]))
			output_data[i] = T(val)
		}
	}

	output := vtl.from_1d[T](output_data, vtl.TensorData{})!
	return output.reshape([int(batch), int(channels), int(out_h), int(out_w)])!
}

// forward performs average pooling on the input tensor
pub fn (layer &AvgPool2DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if layer.device == unsafe { nil } {
		return error('AvgPool2DLayerVulkan: device is nil')
	}
	return avgpool2d_forward_vulkan[T](input, layer.kernel_size, layer.stride, layer.padding,
		layer.device)!
}
