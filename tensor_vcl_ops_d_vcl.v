module vtl

import storage
import vsl.vcl
import vsl.vcl.compute

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
	a_col := vcl_row_to_col_f64(a_f64, m, k)
	b_col := vcl_row_to_col_f64(b_f64, k, n)

	mut dev := vcl.get_default_device()!
	defer {
		dev.release() or {}
	}

	c_col := compute.gemm_vcl(mut dev, a_col, b_col, m, n, k)!
	c_row := vcl_col_to_row_f64(c_col, m, n)
	c := c_row.map(cast[T](it))
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
	c_f64 := compute.add_vec_vcl(mut dev, a_f64, b_f64)!
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
	c_f64 := compute.mul_vec_vcl(mut dev, a_f64, b_f64)!
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
	y_f64 := compute.relu_vcl(mut dev, x_f64)!
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
	y_f64 := compute.sigmoid_vcl(mut dev, x_f64)!
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
	y_f64 := compute.tanh_vcl(mut dev, x_f64)!
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
	y_f64 := compute.add_scalar_vcl(mut dev, x_f64, f64(cast[f64](s)))!
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
	y_f64 := compute.mul_scalar_vcl(mut dev, x_f64, f64(cast[f64](s)))!
	y := y_f64.map(cast[T](it))
	return vcl_tensor_from_array[T](y, t.shape, mut dev)!
}

// --- internal helpers ---

fn vcl_row_to_col_f64(data []f64, rows int, cols int) []f64 {
	mut out := []f64{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r + c * rows] = data[r * cols + c]
		}
	}
	return out
}

fn vcl_col_to_row_f64(data []f64, rows int, cols int) []f64 {
	mut out := []f64{len: data.len}
	for r in 0 .. rows {
		for c in 0 .. cols {
			out[r * cols + c] = data[r + c * rows]
		}
	}
	return out
}

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
