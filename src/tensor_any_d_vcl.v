module vtl

// TensorDataType is a sum type that lists the possible types to be used to define storage
pub type TensorDataType = byte | f32 | f64 | i16 | i64 | i8 | int | u16 | u32 | u64

// AnyTensor is an interface that allows for any tensor to be used in the vtl library
pub interface AnyTensor<T> {
	shape []int
	strides []int
	cpu() &Tensor<T>
	vcl() &VclTensor<T>
	opencl() &VclTensor<T>
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
}
