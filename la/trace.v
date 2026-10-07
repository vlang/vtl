module la

import vsl.la as vsl_la
import vtl

// TraceAxesOptions selects the diagonal and the two axes traced over.
@[params]
pub struct TraceAxesOptions {
pub:
	offset int
	axis1  int
	axis2  int = 1
}

// trace returns the sum of diagonal elements of a square matrix.
pub fn trace[T](t &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	t.assert_square_matrix()!
	m := t.shape[0]
	mat := vsl_la.Matrix.raw(m, m, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.trace(mat)])
}

// trace_axes sums a diagonal of each matrix selected by axis1 and axis2.
// Negative axes count from the end. The default axes are 0 and 1, matching
// numpy.trace; an N-D input returns the remaining dimensions as a batch.
pub fn trace_axes[T](input &vtl.Tensor[T], options TraceAxesOptions) !&vtl.Tensor[f64] {
	if input.rank() < 2 {
		return error('trace_axes requires input with rank at least 2')
	}
	rank := input.rank()
	axis1 := if options.axis1 < 0 { options.axis1 + rank } else { options.axis1 }
	axis2 := if options.axis2 < 0 { options.axis2 + rank } else { options.axis2 }
	if axis1 < 0 || axis1 >= rank || axis2 < 0 || axis2 >= rank {
		return error('trace_axes axis is out of bounds for rank ${rank}')
	}
	if axis1 == axis2 {
		return error('trace_axes requires two distinct axes')
	}
	axis1_size := input.shape[axis1]
	axis2_size := input.shape[axis2]
	mut start_axis1 := 0
	mut start_axis2 := 0
	mut diagonal_size := 0
	if options.offset >= 0 {
		if options.offset < axis2_size {
			start_axis2 = options.offset
			diagonal_size = if axis1_size < axis2_size - start_axis2 {
				axis1_size
			} else {
				axis2_size - start_axis2
			}
		}
	} else if options.offset >= -axis1_size {
		start_axis1 = -options.offset
		diagonal_size = if axis1_size - start_axis1 < axis2_size {
			axis1_size - start_axis1
		} else {
			axis2_size
		}
	}
	mut output_shape := []int{cap: rank - 2}
	for axis, size in input.shape {
		if axis != axis1 && axis != axis2 {
			output_shape << size
		}
	}
	if output_shape.len == 0 {
		output_shape << 1
	}
	mut batch_size := 1
	for dimension in output_shape {
		batch_size *= dimension
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut input_index := []int{len: rank}
	mut output_index := []int{len: output_shape.len}
	for batch in 0 .. batch_size {
		decode_flat_index(batch, output_shape, mut output_index)
		mut output_axis := 0
		for axis in 0 .. rank {
			if axis != axis1 && axis != axis2 {
				input_index[axis] = output_index[output_axis]
				output_axis++
			}
		}
		mut diagonal_sum := 0.0
		for diagonal in 0 .. diagonal_size {
			input_index[axis1] = start_axis1 + diagonal
			input_index[axis2] = start_axis2 + diagonal
			diagonal_sum += f64(input.get[T](input_index))
		}
		result.set_nth(batch, diagonal_sum)
	}
	return result
}

// norm returns the matrix norm of a tensor.
// ord: "F" (Frobenius, default), "1" (column sum), "I" (row sum / infinity).
pub fn norm[T](t &vtl.Tensor[T], ord string) !&vtl.Tensor[f64] {
	if t.rank() != 2 {
		return error('norm: tensor must be 2D (matrix), got rank ${t.rank()}')
	}
	m := t.shape[0]
	n := t.shape[1]
	mat := vsl_la.Matrix.raw(m, n, tensor_to_f64_array[T](t))
	return vtl.from_1d([vsl_la.norm(mat, ord)])
}
