module vtl

// AnyTensor is an interface that allows for any tensor to be used in the vtl library
pub interface AnyTensor[T] {
	shape []int
	strides []int
	cpu() &Tensor[T]
	vcl() !&Tensor[T]
	str() string
	rank() int
	size() int
	is_matrix() bool
	is_square_matrix() bool
	is_vector() bool
	is_row_major() bool
	is_col_major() bool
	is_row_major_contiguous() bool
	is_col_major_contiguous() bool
	is_contiguous() bool
mut:
	memory MemoryFormat
}
