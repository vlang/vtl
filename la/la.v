module la

import vsl.la as vsl_la
import vtl

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
	if a.rank() > 2 || b.rank() > 2 {
		if a.rank() < 2 || b.rank() < 2 || a.shape[a.rank() - 1] != b.shape[b.rank() - 2] {
			return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
		}
		if b.rank() > 2 && a.shape[..a.rank() - 2] != b.shape[..b.rank() - 2] {
			return error('Batch shapes must match for matrix multiplication ${a.shape} and ${b.shape}')
		}

		mut result_shape := a.shape[..a.rank() - 2].clone()
		result_shape << a.shape[a.rank() - 2]
		result_shape << b.shape[b.rank() - 1]
		mut batch_size := 1
		for dimension in a.shape[..a.rank() - 2] {
			batch_size *= dimension
		}
		rows := a.shape[a.rank() - 2]
		inner := a.shape[a.rank() - 1]
		columns := b.shape[b.rank() - 1]
		a_data := a.copy(.row_major).to_array()
		b_data := b.copy(.row_major).to_array()
		mut result_data := []T{len: batch_size * rows * columns}
		for batch in 0 .. batch_size {
			for row in 0 .. rows {
				for column in 0 .. columns {
					mut value := T(0)
					for index in 0 .. inner {
						a_offset := batch * rows * inner + row * inner + index
						b_batch := if b.rank() > 2 { batch } else { 0 }
						b_offset := b_batch * inner * columns + index * columns + column
						value += a_data[a_offset] * b_data[b_offset]
					}
					result_data[batch * rows * columns + row * columns + column] = value
				}
			}
		}
		return vtl.from_array(result_data, result_shape)
	}
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}
	ma := a.copy(.row_major)
	mb := b.copy(.row_major)
	mut dm := vsl_la.Matrix.new[f64](a.shape[0], b.shape[1])
	mam := vsl_la.Matrix.raw(a.shape[0], a.shape[1], tensor_to_f64_array[T](ma))
	mbm := vsl_la.Matrix.raw(b.shape[0], b.shape[1], tensor_to_f64_array[T](mb))
	vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	res := vtl.from_2d[f64](dm.get_deep2())!
	if sizeof(T) == 4 {
		return unsafe { &vtl.Tensor[T](res.as_f32()) }
	}
	return unsafe { &vtl.Tensor[T](res) }
}

fn tensor_to_f64_array[T](t &vtl.Tensor[T]) []f64 {
	return t.as_f64().to_array()
}
