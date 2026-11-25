module vtl

// ufunc_unary applies a unary elementwise operation and allocates its result.
pub fn ufunc_unary[T](input &Tensor[T], operation fn (T) T) !&Tensor[T] {
	mut output := zeros_like[T](input)
	ufunc_unary_into[T](input, output, unsafe { nil }, false, operation)!
	return output
}

// ufunc_unary_into applies a unary operation into a caller-owned output tensor.
// When has_where is true, the mask must match input exactly; false entries are
// left unchanged, which makes aliasing and preallocated masked updates explicit.
pub fn ufunc_unary_into[T](input &Tensor[T], output &Tensor[T], where_mask &Tensor[bool], has_where bool, operation fn (T) T) ! {
	if input.shape != output.shape {
		return error('ufunc_unary_into output shape must match input shape')
	}
	if has_where {
		if where_mask == unsafe { nil } || where_mask.shape != input.shape {
			return error('ufunc_unary_into where mask must match input shape')
		}
	}
	for i in 0 .. input.size() {
		if !has_where || where_mask.get_nth(i) {
			output.set_nth(i, operation(input.get_nth(i)))
		}
	}
}
