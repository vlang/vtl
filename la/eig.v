module la

import math
import math.complex as vcomplex
import vsl.lapack as vsl_lapack
import vtl

// eig computes the general eigenvalues and right eigenvectors of each trailing
// real square matrix. Complex conjugate pairs are returned as complex values;
// eigenvectors are stored in columns. The outputs match NumPy shapes:
// [..., N] and [..., N, N].
pub fn eig[T](input &vtl.Tensor[T]) !(&vtl.Tensor[vcomplex.Complex], &vtl.Tensor[vcomplex.Complex]) {
	return general_eigen[T](input, true)
}

// eigvals returns the general eigenvalues of each trailing real square matrix
// without computing eigenvectors.
pub fn eigvals[T](input &vtl.Tensor[T]) !&vtl.Tensor[vcomplex.Complex] {
	values, _ := general_eigen[T](input, false)!
	return values
}

fn general_eigen[T](input &vtl.Tensor[T], compute_vectors bool) !(&vtl.Tensor[vcomplex.Complex], &vtl.Tensor[vcomplex.Complex]) {
	if input.rank() < 2 {
		return error('eig requires input with rank at least 2')
	}
	n := input.shape[input.rank() - 1]
	if input.shape[input.rank() - 2] != n {
		return error('eig requires square matrices')
	}
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut values_shape := batch_shape.clone()
	values_shape << n
	mut vectors_shape := batch_shape.clone()
	vectors_shape << n
	vectors_shape << n
	mut values := []vcomplex.Complex{len: batch_count * n}
	mut vectors := if compute_vectors {
		[]vcomplex.Complex{len: batch_count * n * n}
	} else {
		[]vcomplex.Complex{}
	}
	if n == 0 || batch_count == 0 {
		return vtl.from_array[vcomplex.Complex](values, values_shape)!, vtl.from_array[vcomplex.Complex](vectors,
			if compute_vectors { vectors_shape } else { [0] })!
	}
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix := [][]f64{len: n, init: []f64{len: n}}
		for row in 0 .. n {
			input_index[input.rank() - 2] = row
			for column in 0 .. n {
				input_index[input.rank() - 1] = column
				value := f64(input.get[T](input_index))
				if math.is_nan(value) || math.is_inf(value, 0) {
					return error('eig requires finite matrix values')
				}
				matrix[row][column] = value
			}
		}
		jobvr := if compute_vectors {
			vsl_lapack.RightEigenVectorsJob.right_ev_compute
		} else {
			vsl_lapack.RightEigenVectorsJob.right_ev_none
		}
		real_parts, imaginary_parts, _, right_vectors := vsl_lapack.geev(matrix,
			vsl_lapack.LeftEigenVectorsJob.left_ev_none, jobvr) or {
			return error('eig: ${err}')
		}
		for component in 0 .. n {
			values[batch * n + component] = vcomplex.Complex{
				re: real_parts[component]
				im: imaginary_parts[component]
			}
		}
		if compute_vectors {
			for component in 0 .. n {
				if imaginary_parts[component] < 0 {
					continue
				}
				for row in 0 .. n {
					if imaginary_parts[component] > 0 {
						real_value := right_vectors[row][component]
						imaginary_value := right_vectors[row][component + 1]
						vectors[(batch * n + row) * n + component] = vcomplex.Complex{
							re: real_value
							im: imaginary_value
						}
						vectors[(batch * n + row) * n + component + 1] = vcomplex.Complex{
							re: real_value
							im: -imaginary_value
						}
					} else {
						vectors[(batch * n + row) * n + component] = vcomplex.Complex{
							re: right_vectors[row][component]
							im: 0
						}
					}
				}
			}
		}
	}
	values_tensor := vtl.from_array[vcomplex.Complex](values, values_shape)!
	if !compute_vectors {
		return values_tensor, vtl.empty[vcomplex.Complex]([0])
	}
	return values_tensor, vtl.from_array[vcomplex.Complex](vectors, vectors_shape)!
}
