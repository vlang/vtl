module la

import vsl.la as vsl_la
import vtl
import vtl.storage

// dot exposes this operation as part of the public API.
pub fn dot[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if !a.is_vector() || !b.is_vector() {
		return error('Tensors must be one dimensional')
	} else if a.size != b.size {
		return error('Tensors must have the same shape')
	}
	res := vsl_la.vector_dot(tensor_to_f64_array[T](a), tensor_to_f64_array[T](b))
	return vtl.from_1d([res])
}

// det exposes this operation as part of the public API.
pub fn det[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	m := t.shape[0]
	n := t.shape[1]
	mat := vsl_la.Matrix.raw(m, n, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.matrix_det(mat)])
}

// inv exposes this operation as part of the public API.
pub fn inv[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	mut colmajort := t.copy(.col_major)
	mut ret_m := vsl_la.Matrix.new[f64](colmajort.shape[0], colmajort.shape[1])
	mut colmajorm := vsl_la.Matrix.raw(colmajort.shape[0], colmajort.shape[1],
		tensor_to_f64_array[T](colmajort))
	vsl_la.matrix_inv(mut ret_m, mut colmajorm, true)
	return vtl.from_2d[f64](ret_m.get_deep2())
}

// matmul exposes this operation as part of the public API.
pub fn matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if a.rank() < 2 || b.rank() < 2 {
		if a.rank() == 0 || b.rank() == 0 {
			return error('Matrix multiplication requires tensors with rank at least one')
		}
		a_is_vector := a.rank() == 1
		b_is_vector := b.rank() == 1
		mut a_shape := a.shape.clone()
		mut b_shape := b.shape.clone()
		if a_is_vector {
			a_shape = [1, a.shape[0]]
		}
		if b_is_vector {
			b_shape = [b.shape[0], 1]
		}
		promoted_a := a.reshape(a_shape)!
		promoted_b := b.reshape(b_shape)!
		result := matmul[T](promoted_a, promoted_b)!
		mut result_shape := result.shape.clone()
		if a_is_vector {
			result_shape.delete(result_shape.len - 2)
		}
		if b_is_vector {
			result_shape.delete(result_shape.len - 1)
		}
		return result.reshape(result_shape)
	}
	$if T is f32 {
		return matmul_f32(a, b)
	}
	if a.rank() > 2 || b.rank() > 2 {
		if a.rank() < 2 || b.rank() < 2 || a.shape[a.rank() - 1] != b.shape[b.rank() - 2] {
			return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
		}
		a_batch_shape := a.shape[..a.rank() - 2]
		b_batch_shape := b.shape[..b.rank() - 2]
		batch_shape := matmul_broadcast_shape(a_batch_shape, b_batch_shape) or {
			return error('Batch shapes ${a_batch_shape} and ${b_batch_shape} cannot be broadcast for matrix multiplication ${a.shape} and ${b.shape}')
		}
		mut result_shape := batch_shape.clone()
		result_shape << a.shape[a.rank() - 2]
		result_shape << b.shape[b.rank() - 1]
		mut batch_size := 1
		for dimension in batch_shape {
			batch_size *= dimension
		}
		rows := a.shape[a.rank() - 2]
		inner := a.shape[a.rank() - 1]
		columns := b.shape[b.rank() - 1]
		a_data := tensor_to_f64_array[T](a)
		b_data := tensor_to_f64_array[T](b)
		mut result_data := []T{len: batch_size * rows * columns}
		for batch in 0 .. batch_size {
			a_batch := matmul_broadcast_offset(batch, batch_shape, a_batch_shape)
			b_batch := matmul_broadcast_offset(batch, batch_shape, b_batch_shape)
			a_start := a_batch * rows * inner
			b_start := b_batch * inner * columns
			mam := vsl_la.Matrix.raw(rows, inner, a_data[a_start..a_start + rows * inner])
			mbm := vsl_la.Matrix.raw(inner, columns, b_data[b_start..b_start + inner * columns])
			mut dm := vsl_la.Matrix.new[f64](rows, columns)
			vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
			for i, value in dm.data {
				result_data[batch * rows * columns + i] = vtl.cast[T](value)
			}
		}
		return vtl.from_array(result_data, result_shape)
	}
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}
	mut dm := vsl_la.Matrix.new[f64](a.shape[0], b.shape[1])
	mam := vsl_la.Matrix.raw(a.shape[0], a.shape[1], tensor_to_f64_array[T](a))
	mbm := vsl_la.Matrix.raw(b.shape[0], b.shape[1], tensor_to_f64_array[T](b))
	vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	res := &vtl.Tensor[f64]{
		data:    &storage.CpuStorage[f64]{
			data: dm.data
		}
		memory:  .row_major
		size:    a.shape[0] * b.shape[1]
		shape:   [a.shape[0], b.shape[1]]
		strides: [b.shape[1], 1]
	}
	$if T is f32 {
		return unsafe { &vtl.Tensor[T](res.as_f32()) }
	} $else $if T is f64 {
		return unsafe { &vtl.Tensor[T](res) }
	} $else {
		result_data := res.to_array().map(vtl.cast[T](it))
		return vtl.from_array[T](result_data, [a.shape[0], b.shape[1]])
	}
}

fn matmul_f32(a &vtl.Tensor[f32], b &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if a.rank() < 2 || b.rank() < 2 || a.shape[a.rank() - 1] != b.shape[b.rank() - 2] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}
	a_batch_shape := a.shape[..a.rank() - 2]
	b_batch_shape := b.shape[..b.rank() - 2]
	batch_shape := matmul_broadcast_shape(a_batch_shape, b_batch_shape) or {
		return error('Batch shapes ${a_batch_shape} and ${b_batch_shape} cannot be broadcast for matrix multiplication ${a.shape} and ${b.shape}')
	}
	mut batch_size := 1
	for dimension in batch_shape {
		batch_size *= dimension
	}
	rows := a.shape[a.rank() - 2]
	inner := a.shape[a.rank() - 1]
	columns := b.shape[b.rank() - 1]
	mut result_shape := batch_shape.clone()
	result_shape << rows
	result_shape << columns
	a_data := tensor_to_f32_array(a)
	b_data := tensor_to_f32_array(b)
	mut result_data := []f32{len: batch_size * rows * columns}
	for batch in 0 .. batch_size {
		a_batch := matmul_broadcast_offset(batch, batch_shape, a_batch_shape)
		b_batch := matmul_broadcast_offset(batch, batch_shape, b_batch_shape)
		a_start := a_batch * rows * inner
		b_start := b_batch * inner * columns
		result_start := batch * rows * columns
		result_end := result_start + rows * columns
		vsl_la.matrix_matrix_mul_f32(mut result_data[result_start..result_end], rows, columns, inner, 1,
			a_data[a_start..a_start + rows * inner],
			b_data[b_start..b_start + inner * columns])
	}
	return vtl.from_array[f32](result_data, result_shape)
}

fn matmul_broadcast_shape(a []int, b []int) ![]int {
	rank := if a.len > b.len { a.len } else { b.len }
	mut result := []int{len: rank, init: 1}
	for i in 0 .. rank {
		a_index := i - (rank - a.len)
		b_index := i - (rank - b.len)
		a_dim := if a_index >= 0 { a[a_index] } else { 1 }
		b_dim := if b_index >= 0 { b[b_index] } else { 1 }
		if a_dim == b_dim {
			result[i] = a_dim
		} else if a_dim == 1 {
			result[i] = b_dim
		} else if b_dim == 1 {
			result[i] = a_dim
		} else {
			return error('shapes ${a} and ${b} are not broadcastable')
		}
	}
	return result
}

fn matmul_broadcast_offset(output_batch int, output_shape []int, input_shape []int) int {
	mut remaining := output_batch
	mut input_offset := 0
	mut input_stride := 1
	for axis := output_shape.len - 1; axis >= 0; axis-- {
		coordinate := remaining % output_shape[axis]
		remaining /= output_shape[axis]
		input_axis := axis - (output_shape.len - input_shape.len)
		if input_axis >= 0 {
			if input_shape[input_axis] != 1 {
				input_offset += coordinate * input_stride
			}
			input_stride *= input_shape[input_axis]
		}
	}
	return input_offset
}

fn tensor_to_f64_array[T](t &vtl.Tensor[T]) []f64 {
	$if T is f64 {
		if t.is_row_major_contiguous() {
			return t.data.data[..t.size]
		}
	}
	return t.as_f64().to_array()
}

fn tensor_to_f32_array(t &vtl.Tensor[f32]) []f32 {
	if t.is_row_major_contiguous() {
		return t.data.data[..t.size]
	}
	return t.to_array()
}
