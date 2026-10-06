module core

import vtl

fn test_copy_col_major_preserves_layout() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mut result := t.copy(.col_major)
	result.ensure_memory()

	assert result.is_col_major()
	assert result.is_col_major_contiguous()
	assert result.strides == [1, 2]
	assert result.array_equal(t)
}
