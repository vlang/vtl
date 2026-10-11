module layers

import vtl

// linear_add_bias adds the broadcast bias directly to a fresh GEMM result when
// the common row-bias layout is contiguous; other valid broadcasts use add.
@[direct_array_access]
fn linear_add_bias[T](mut output &vtl.Tensor[T], bias &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if output.rank() >= 2 && bias.rank() == 2 && bias.shape[0] == 1 {
		width := output.shape[output.rank() - 1]
		if width > 0 && bias.shape[1] == width && output.is_row_major_contiguous()
			&& bias.is_row_major_contiguous() && output.data.data.len == output.size
			&& bias.data.data.len == bias.size {
			for i in 0 .. output.size {
				output.data.data[i] = output.data.data[i] + bias.data.data[i % width]
			}
			return output
		}
	}
	return output.add[T](bias)
}
