module layers

import vtl
import vsl.vulkan
import vtl.storage

// conv2d_forward_vulkan implements Conv2D forward pass on Vulkan via im2col + GEMM.
// input: [batch, in_ch, H, W]
// weight: [out_ch, in_ch, k_h, k_w]
// bias: [out_ch] or [1, out_ch]
// Returns: [batch, out_ch, out_H, out_W]
pub fn conv2d_forward_vulkan[T](
	input       &vtl.Tensor[T],
	weight      &vtl.Tensor[T],
	bias        &vtl.Tensor[T],
	kernel_size []int,
	config      Conv2DConfig
) !&vtl.Tensor[T] {
	// Only f32 supported for now (Vulkan GEMM is f32 native)
	$if T !is f32 {
		return error('conv2d_forward_vulkan: only f32 supported')
	}

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

	// Create Vulkan device
	mut dev := vulkan.new_device() or { return error('conv2d_forward_vulkan: no Vulkan device available') }
	defer { dev.release() }

	// Allocate GPU buffers
	im2col_size := batch * out_h * out_w * in_ch * k_h * k_w
	mut im2col_buf := dev.buffer(vulkan.DeviceSize(im2col_size * 4))!
	defer { im2col_buf.release() }

	// Upload input to GPU
	input_size := batch * in_ch * in_h * in_w
	mut input_bytes := []u8{len: int(input_size * 4)}
	for i in 0 .. int(input_size) {
		val := f32(input.get_nth(i))
		unsafe { *(&f32(&input_bytes[i * 4])) = val }
	}
	mut input_buf := dev.buffer(vulkan.DeviceSize(input_size * 4))!
	defer { input_buf.release() }
	input_buf.load(input_bytes)!

	// Im2col transform on GPU: input → im2col_buf
	// Result shape: [batch * out_h * out_w, in_ch * k_h * k_w]
	vulkan.im2col(dev, im2col_buf, input_buf, batch, in_ch, in_h, in_w, k_h, k_w,
		out_h, out_w, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w)!

	// Prepare weight matrix: [out_ch, in_ch, k_h, k_w] → [out_ch, in_ch*k_h*k_w]
	// Then transpose to [in_ch*k_h*k_w, out_ch] for GEMM B^T layout
	weight_cols := int(in_ch * k_h * k_w)
	weight_rows := int(out_ch)
	mut weight_t_data := []f32{len: weight_rows * weight_cols}
	for r in 0 .. weight_rows {
		for c in 0 .. weight_cols {
			idx := r * weight_cols + c
			weight_t_data[c * weight_rows + r] = f32(weight.get_nth(idx))
		}
	}
	mut weight_bytes := []u8{len: weight_t_data.len * 4}
	for i, val in weight_t_data {
		unsafe { *(&f32(&weight_bytes[i * 4])) = val }
	}
	mut weight_buf := dev.buffer(vulkan.DeviceSize(weight_bytes.len))!
	defer { weight_buf.release() }
	weight_buf.load(weight_bytes)!

	// GEMM: im2col @ weight^T
	// im2col: [m, k] = [batch*out_h*out_w, in_ch*k_h*k_w]
	// weight^T: [k, n] = [in_ch*k_h*k_w, out_ch]
	// result: [m, n] = [batch*out_h*out_w, out_ch]
	m := batch * out_h * out_w
	n := out_ch
	k := in_ch * k_h * k_w
	mut gemm_out_buf := dev.buffer(vulkan.DeviceSize(m * n * 4))!
	defer { gemm_out_buf.release() }

	// Execute GEMM on GPU
	vulkan.gemm(dev, gemm_out_buf, im2col_buf, weight_buf, m, n, k)!

	// Download result from GPU
	mut result_bytes := []u8{len: int(m * n * 4)}
	gemm_out_buf.store(mut result_bytes)!
	mut result_data := []T{len: int(m * n)}
	for i in 0 .. int(m * n) {
		unsafe { result_data[i] = T(*(&f32(&result_bytes[i * 4]))) }
	}

	// Add bias: [batch*out_h*out_w, out_ch] + [1, out_ch]
	bias_flat := bias.flatten()
	for i in 0 .. int(m) {
		for j in 0 .. int(n) {
			idx := i * int(n) + j
			result_data[idx] = T(f32(result_data[idx]) + f32(bias_flat.get_nth(j)))
		}
	}

	// Reshape to [batch, out_ch, out_h, out_w]
	mut result_1d := vtl.from_1d[T](result_data, vtl.TensorData{}) or { return err }
	return result_1d.reshape([int(batch), int(out_ch), int(out_h), int(out_w)])!
}
