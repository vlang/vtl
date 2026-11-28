module la

import vsl.la as vsl_la
import vsl.blas
import vtl
import vtl.storage
import math
import math.complex as vcomplex

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
	if t.rank() < 2 {
		return error('det requires input with rank at least 2')
	}
	rows := t.shape[t.rank() - 2]
	columns := t.shape[t.rank() - 1]
	if rows != columns {
		return error('det requires square matrices, got ${rows}x${columns}')
	}
	signs, logabsdets := slogdet(t)!
	output_shape := if t.rank() == 2 { [1] } else { t.shape[..t.rank() - 2].clone() }
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	for i, sign in signs.to_array() {
		value := if sign == 0 { 0.0 } else { sign * math.exp(logabsdets.get_nth(i)) }
		result.set_nth(i, value)
	}
	return result
}

// inv exposes this operation as part of the public API.
pub fn inv[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if t.rank() < 2 {
		return error('inv requires input with rank at least 2')
	}
	rows := t.shape[t.rank() - 2]
	columns := t.shape[t.rank() - 1]
	if rows != columns {
		return error('inv requires square matrices, got ${rows}x${columns}')
	}
	batch_shape := t.shape[..t.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut result := vtl.empty[f64](t.shape, memory: .row_major)
	mut input_index := []int{len: t.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[t.rank() - 2] = row
			for column in 0 .. columns {
				input_index[t.rank() - 1] = column
				matrix[row * columns + column] = f64(t.get[T](input_index))
			}
		}
		inverse := invert_square_matrix(matrix, rows)!
		start := batch * rows * columns
		for i, value in inverse {
			result.set_nth(start + i, value)
		}
	}
	return result
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
	$if T is vcomplex.Complex {
		return matmul_complex(a, b)
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

fn array_all_finite(values []f64) bool {
	for value in values {
		if !math.is_finite(value) {
			return false
		}
	}
	return true
}

fn sums_preserve_operands(left []f64, right []f64, sum []f64) bool {
	for i in 0 .. sum.len {
		if !math.is_finite(sum[i]) {
			return false
		}
		// The three-product form is only equivalent if (Re+Im) did not
		// round/absorb either operand; otherwise fall back to 4 products.
		if (sum[i] - left[i]) != right[i] || (sum[i] - right[i]) != left[i] {
			return false
		}
	}
	return true
}

fn three_product_results_finite(p1 []f64, p2 []f64, p3 []f64) bool {
	for i in 0 .. p1.len {
		re := p1[i] - p2[i]
		im := p3[i] - p1[i] - p2[i]
		if !math.is_finite(re) || !math.is_finite(im) {
			return false
		}
	}
	return true
}

// complex_term_bound bounds the operand-term magnitudes summed into the
// three products for each entry, rather than the (possibly cancelled)
// outputs. row_abs_sums[i] = sum_k (|a_re|+|a_im|) and
// col_abs_maxs[j] = max_k (|b_re|+|b_im|) give, for entry (i, j):
//   sum_k |a_re*b_re| + sum_k |a_im*b_im| + sum_k |a_sum*b_sum| <= 2*row*col
// The (inner+2)*eps factor covers the at most inner roundings in each
// dgemm summation and the two subtractions forming Im (inner*eps << 1
// for any matrix that fits in memory).
fn complex_term_bound(a_re []f64, a_im []f64, b_re []f64, b_im []f64, rows int, columns int, inner int) []f64 {
	eps := 2.220446049250313e-16
	mut row_abs_sums := []f64{len: rows}
	for i in 0 .. rows {
		mut total := 0.0
		for k in 0 .. inner {
			total += math.abs(a_re[i * inner + k]) + math.abs(a_im[i * inner + k])
		}
		row_abs_sums[i] = total
	}
	mut col_abs_maxs := []f64{len: columns}
	for j in 0 .. columns {
		mut largest := 0.0
		for k in 0 .. inner {
			magnitude := math.abs(b_re[k * columns + j]) + math.abs(b_im[k * columns + j])
			if magnitude > largest {
				largest = magnitude
			}
		}
		col_abs_maxs[j] = largest
	}
	factor := 2.0 * f64(inner + 2) * eps
	mut bound := []f64{len: rows * columns}
	for i in 0 .. rows {
		for j in 0 .. columns {
			bound[i * columns + j] = factor * row_abs_sums[i] * col_abs_maxs[j]
		}
	}
	return bound
}

// three_product_accurate reports whether both components of the
// three-product result keep enough significant bits. Re = p1-p2 and
// Im = p3-p1-p2 each subtract separately summed full-scale products, so a
// small component can be absorbed even when every value is finite and the
// (Re+Im) sums preserved their operands; the term-wise form avoids this.
// term_bound[i] bounds the operand-term magnitudes behind the products
// (see complex_term_bound): the outputs can cancel to almost zero while
// their terms stay huge, so output magnitudes alone cannot reveal the
// loss. Accept only when the estimated error is at most 1e-8 of BOTH
// components; otherwise use the term-wise form.
fn three_product_accurate(p1 []f64, p2 []f64, p3 []f64, term_bound []f64) bool {
	for i in 0 .. p1.len {
		re := p1[i] - p2[i]
		im := p3[i] - p1[i] - p2[i]
		if term_bound[i] > 1e-8 * math.abs(re) || term_bound[i] > 1e-8 * math.abs(im) {
			return false
		}
	}
	return true
}

fn matmul_complex(a &vtl.Tensor[vcomplex.Complex], b &vtl.Tensor[vcomplex.Complex]) !&vtl.Tensor[vcomplex.Complex] {
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
	if rows == 0 || columns == 0 || batch_size == 0 {
		return vtl.from_array[vcomplex.Complex]([]vcomplex.Complex{}, result_shape)
	}
	a_data := a.to_array()
	b_data := b.to_array()
	mut a_re := []f64{len: a_data.len}
	mut a_im := []f64{len: a_data.len}
	mut b_re := []f64{len: b_data.len}
	mut b_im := []f64{len: b_data.len}
	for i, value in a_data {
		a_re[i] = value.re
		a_im[i] = value.im
	}
	for i, value in b_data {
		b_re[i] = value.re
		b_im[i] = value.im
	}
	mut result_data := []vcomplex.Complex{len: batch_size * rows * columns}
	mut a_sum := []f64{len: rows * inner}
	mut b_sum := []f64{len: inner * columns}
	mut p1 := []f64{len: rows * columns}
	mut p2 := []f64{len: rows * columns}
	mut p3 := []f64{len: rows * columns}
	for batch in 0 .. batch_size {
		if rows == 0 || columns == 0 || inner == 0 {
			continue
		}
		a_batch := matmul_broadcast_offset(batch, batch_shape, a_batch_shape)
		b_batch := matmul_broadcast_offset(batch, batch_shape, b_batch_shape)
		a_start := a_batch * rows * inner
		b_start := b_batch * inner * columns
		result_start := batch * rows * columns
		for i in 0 .. rows * inner {
			a_sum[i] = a_re[a_start + i] + a_im[a_start + i]
		}
		for i in 0 .. inner * columns {
			b_sum[i] = b_re[b_start + i] + b_im[b_start + i]
		}
		// Three real products: p1 = Re(A)Re(B), p2 = Im(A)Im(B),
		// p3 = (Re(A)+Im(A))(Re(B)+Im(B)); Re(C) = p1-p2, Im(C) = p3-p1-p2.
		blas.dgemm(.no_trans, .no_trans, rows, columns, inner, 1.0, a_re[a_start..], inner, b_re[b_start..], columns, 0.0, mut p1, columns)
		blas.dgemm(.no_trans, .no_trans, rows, columns, inner, 1.0, a_im[a_start..], inner, b_im[b_start..], columns, 0.0, mut p2, columns)
		mut used_three_product := false
		if array_all_finite(a_sum) && array_all_finite(b_sum) && array_all_finite(p1) && array_all_finite(p2) && sums_preserve_operands(a_re[a_start..a_start + rows * inner], a_im[a_start..a_start + rows * inner], a_sum) && sums_preserve_operands(b_re[b_start..b_start + inner * columns], b_im[b_start..b_start + inner * columns], b_sum) {
			blas.dgemm(.no_trans, .no_trans, rows, columns, inner, 1.0, a_sum, inner, b_sum, columns, 0.0, mut p3, columns)
			term_bound := complex_term_bound(a_re[a_start..a_start + rows * inner], a_im[a_start..a_start + rows * inner], b_re[b_start..b_start + inner * columns], b_im[b_start..b_start + inner * columns], rows, columns, inner)
			if array_all_finite(p3) && three_product_results_finite(p1, p2, p3) && three_product_accurate(p1,
				p2, p3, term_bound)
			{
				for i in 0 .. rows * columns {
					result_data[result_start + i] = vcomplex.Complex{ re: p1[i] - p2[i], im: p3[i] - p1[i] - p2[i] }
				}
				used_three_product = true
			}
		}
		if !used_three_product {
			// Accumulate each complex product term by term. p1 and p2 are
			// separately summed real products, so reusing them would absorb
			// small real parts; two separately summed cross products can
			// absorb a small imaginary part into a large one. This form is
			// the same one used before this PR, including when p1 or p2
			// overflowed to Inf.
			for i in 0 .. rows {
				for j in 0 .. columns {
					mut re := 0.0
					mut im := 0.0
					for k in 0 .. inner {
						ar := a_re[a_start + i * inner + k]
						ai := a_im[a_start + i * inner + k]
						br := b_re[b_start + k * columns + j]
						bi := b_im[b_start + k * columns + j]
						re += ar * br - ai * bi
						im += ar * bi + ai * br
					}
					result_data[result_start + i * columns + j] = vcomplex.Complex{ re: re, im: im }
				}
			}
		}
	}
	return vtl.from_array[vcomplex.Complex](result_data, result_shape)
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
