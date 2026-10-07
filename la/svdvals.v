module la

import vtl

// svdvals returns the singular values of every trailing real matrix,
// descending within each matrix. The output shape is [..., min(M, N)].
// Non-finite inputs and inputs below rank two return errors.
pub fn svdvals[T](input &vtl.Tensor[T]) !&vtl.Tensor[f64] {
	if input.rank() < 2 {
		return error('svdvals requires input with rank at least 2')
	}
	rows := input.shape[input.rank() - 2]
	columns := input.shape[input.rank() - 1]
	value_count := if rows < columns { rows } else { columns }
	batch_shape := input.shape[..input.rank() - 2].clone()
	mut batch_count := 1
	for dimension in batch_shape {
		batch_count *= dimension
	}
	mut output_shape := batch_shape.clone()
	output_shape << value_count
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	if value_count == 0 {
		return result
	}
	mut input_index := []int{len: input.rank()}
	for batch in 0 .. batch_count {
		decode_matrix_batch(batch, batch_shape, mut input_index)
		mut matrix_data := []f64{len: rows * columns}
		for row in 0 .. rows {
			input_index[input.rank() - 2] = row
			for column in 0 .. columns {
				input_index[input.rank() - 1] = column
				matrix_data[column * rows + row] = f64(input.get[T](input_index))
			}
		}
		singular_values := matrix_singular_values(matrix_data, rows, columns)!
		for i, value in singular_values {
			result.set_nth(batch * value_count + i, value)
		}
	}
	return result
}
