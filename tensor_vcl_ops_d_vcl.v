module vtl

import storage
import vsl.vcl
import vsl.compute as vsl_compute

// matmul returns the matrix product of two VclTensors via OpenCL GEMM.
// Both tensors must be 2-D. Returns a new VclTensor on the same device.
pub fn (a &VclTensor[T]) matmul[T](b &VclTensor[T]) !&VclTensor[T] {
	if !a.is_matrix() || !b.is_matrix() {
		return error('VclTensor.matmul: both tensors must be 2-D')
	}
	if a.shape[1] != b.shape[0] {
		return error('VclTensor.matmul: incompatible shapes ${a.shape} x ${b.shape}')
	}

	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	a_arr := a.data.to_array()!
	b_arr := b.data.to_array()!

	// compute.gemm_vcl uses column-major layout; convert from row-major
	a_f64 := a_arr.map(f64(cast[f64](it)))
	b_f64 := b_arr.map(f64(cast[f64](it)))

	mut cctx := vsl_compute.new_context(.vcl)
	c_row := vsl_compute.gemm(cctx, a_f64, b_f64, m, n, k)!
	c := c_row.map(cast[T](it))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	return vcl_tensor_from_array[T](c, [m, n], mut dev)!
}

// add returns elementwise addition of two VclTensors (same shape).
pub fn (a &VclTensor[T]) add[T](b &VclTensor[T]) !&VclTensor[T] {
	if a.shape != b.shape {
		return error('VclTensor.add: shape mismatch ${a.shape} vs ${b.shape}')
	}
	a_arr := a.data.to_array()!
	b_arr := b.data.to_array()!
	a_f64 := a_arr.map(f64(cast[f64](it)))
	b_f64 := b_arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	c_f64 := vsl_compute.add_vec(cctx, a_f64, b_f64)!
	c := c_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](c, a.shape, mut dev)!
}

// multiply returns elementwise multiplication of two VclTensors (same shape).
pub fn (a &VclTensor[T]) multiply[T](b &VclTensor[T]) !&VclTensor[T] {
	if a.shape != b.shape {
		return error('VclTensor.multiply: shape mismatch ${a.shape} vs ${b.shape}')
	}
	a_arr := a.data.to_array()!
	b_arr := b.data.to_array()!
	a_f64 := a_arr.map(f64(cast[f64](it)))
	b_f64 := b_arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	c_f64 := vsl_compute.mul_vec(cctx, a_f64, b_f64)!
	c := c_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](c, a.shape, mut dev)!
}

// relu applies ReLU activation on a VclTensor element-wise.
pub fn (t &VclTensor[T]) relu[T]() !&VclTensor[T] {
	arr := t.data.to_array()!
	x_f64 := arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	y_f64 := vsl_compute.relu(cctx, x_f64)!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// sigmoid applies sigmoid activation on a VclTensor element-wise.
pub fn (t &VclTensor[T]) sigmoid[T]() !&VclTensor[T] {
	arr := t.data.to_array()!
	x_f64 := arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	y_f64 := vsl_compute.sigmoid(cctx, x_f64)!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// tanh_act applies tanh activation on a VclTensor element-wise.
pub fn (t &VclTensor[T]) tanh_act[T]() !&VclTensor[T] {
	arr := t.data.to_array()!
	x_f64 := arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	y_f64 := vsl_compute.tanh(cctx, x_f64)!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// add_scalar adds a scalar to every element of a VclTensor.
pub fn (t &VclTensor[T]) add_scalar[T](s T) !&VclTensor[T] {
	arr := t.data.to_array()!
	x_f64 := arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	y_f64 := vsl_compute.add_scalar(cctx, x_f64, f64(cast[f64](s)))!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// mul_scalar multiplies every element of a VclTensor by a scalar.
pub fn (t &VclTensor[T]) mul_scalar[T](s T) !&VclTensor[T] {
	arr := t.data.to_array()!
	x_f64 := arr.map(f64(cast[f64](it)))
	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut cctx := vsl_compute.new_context(.vcl)
	y_f64 := vsl_compute.mul_scalar(cctx, x_f64, f64(cast[f64](s)))!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// --- internal helpers ---

fn vcl_tensor_from_array[T](data []T, shape []int, mut dev vcl.Device) !&VclTensor[T] {
	mut vec := dev.vector[T](data.len)!
	err := <-vec.load(data)
	if err !is none {
		return err
	}
	sz := size_from_shape(shape)
	strides := strides_from_shape(shape, .row_major)
	return &VclTensor[T]{
		data:    &storage.VclStorage[T]{
			data: vec
		}
		memory:  .row_major
		size:    sz
		shape:   shape
		strides: strides
	}
}
