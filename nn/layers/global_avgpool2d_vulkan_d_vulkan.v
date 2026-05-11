module layers

import vtl
import vsl.vulkan

// GlobalAvgPool2DLayerVulkan implements global average pooling on GPU
// Reduces spatial dimensions (H×W) to 1×1 per channel
pub struct GlobalAvgPool2DLayerVulkan[T] {
pub mut:
	device &vulkan.Device = unsafe { nil }
}

// global_avgpool2d_forward_vulkan performs global average pooling on GPU.
// input: [batch, channels, height, width]
// output: [batch, channels, 1, 1]
pub fn global_avgpool2d_forward_vulkan[T](input &vtl.Tensor[T], dev &vulkan.Device) !&vtl.Tensor[T] {
	// Extract input dimensions (NCHW layout)
	batch := u32(input.shape[0])
	channels := u32(input.shape[1])
	height := u32(input.shape[2])
	width := u32(input.shape[3])

	// Allocate GPU buffers
	input_size := batch * channels * height * width * 4 // f32 = 4 bytes
	output_size := batch * channels * 1 * 1 * 4

	mut input_buf := dev.buffer(vulkan.DeviceSize(input_size))!
	defer { input_buf.release() }
	mut output_buf := dev.buffer(vulkan.DeviceSize(output_size))!
	defer { output_buf.release() }

	// Upload input data (convert to f32 and flatten in NCHW order)
	mut input_bytes := []u8{len: int(input_size)}
	for i := 0; i < int(batch * channels * height * width); i++ {
		b := i / int(channels * height * width)
		c := (i / int(height * width)) % int(channels)
		h := (i / int(width)) % int(height)
		w := i % int(width)
		val := f32(input.get([b, c, h, w]))
		unsafe {
			*(&f32(&input_bytes[i * 4])) = val
		}
	}
	input_buf.load(input_bytes)!

	// Run global avgpool2d kernel
	vulkan.global_avgpool2d(dev, output_buf, input_buf, batch, channels, height, width)!

	// Download output
	mut output_bytes := []u8{len: int(output_size)}
	output_bytes = output_buf.store(mut output_bytes)!

	// Convert to tensor
	mut output_data := []T{len: int(batch * channels)}
	for i in 0 .. int(batch * channels) {
		unsafe {
			val := *(&f32(&output_bytes[i * 4]))
			output_data[i] = T(val)
		}
	}

	output := vtl.from_1d[T](output_data, vtl.TensorData{})!
	return output.reshape([int(batch), int(channels), 1, 1])!
}

// forward performs global average pooling on the input tensor
pub fn (layer &GlobalAvgPool2DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if layer.device == unsafe { nil } {
		return error('GlobalAvgPool2DLayerVulkan: device is nil')
	}
	return global_avgpool2d_forward_vulkan[T](input, layer.device)!
}
