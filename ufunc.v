module vtl

pub fn ufunc_unary[T](input &Tensor[T], operation fn (T) T) !&Tensor[T] {
	mut output := zeros_like[T](input)
	ufunc_unary_into[T](input, output, unsafe { nil }, false, operation)!
	return output
}

// ufunc_unary_into preserves masked-out values in the caller-owned output.
pub fn ufunc_unary_into[T](input &Tensor[T], output &Tensor[T], where_mask &Tensor[bool], has_where bool, operation fn (T) T) ! {
	if input.shape != output.shape {
		return error('ufunc_unary_into output shape must match input shape')
	}
	if has_where && (where_mask == unsafe { nil } || where_mask.shape != input.shape) {
		return error('ufunc_unary_into where mask must match input shape')
	}
	for i in 0 .. input.size() {
		if !has_where || where_mask.get_nth(i) {
			output.set_nth(i, operation(input.get_nth(i)))
		}
	}
}

pub fn ufunc_binary[T](left &Tensor[T], right &Tensor[T], operation fn (T, T) T) !&Tensor[T] {
	mut output := zeros_like[T](left)
	ufunc_binary_into[T](left, right, output, unsafe { nil }, false, operation)!
	return output
}

pub fn ufunc_binary_into[T](left &Tensor[T], right &Tensor[T], output &Tensor[T], where_mask &Tensor[bool], has_where bool, operation fn (T, T) T) ! {
	if left.shape != right.shape || left.shape != output.shape {
		return error('ufunc_binary_into operands and output must have identical shapes')
	}
	if has_where && (where_mask == unsafe { nil } || where_mask.shape != left.shape) {
		return error('ufunc_binary_into where mask must match operand shape')
	}
	for i in 0 .. left.size() {
		if !has_where || where_mask.get_nth(i) {
			output.set_nth(i, operation(left.get_nth(i), right.get_nth(i)))
		}
	}
}
