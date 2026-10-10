module vtl

// add adds two tensors elementwise

// add exposes this operation as part of the public API.

// add exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) add[T](b &Tensor[T]) !&Tensor[T] {
	return a.map_pair_values[T](b, fn [T](a T, b T) T {
		$if T is bool {
			return td[T](a).bool() || td[T](b).bool()
		} $else $if T is string {
			return '${a.str()}${b.str()}'
		} $else {
			return a + b
		}
	})
}

// add adds a scalar to a tensor elementwise

// add_scalar exposes this operation as part of the public API.

// add_scalar exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) add_scalar[T](scalar T) !&Tensor[T] {
	return a.map_values(fn [scalar] [T](x T) T {
		$if T is bool {
			return td[T](x).bool() || td[T](scalar).bool()
		} $else $if T is string {
			return '${x.str()}${scalar.str()}'
		} $else {
			return x + scalar
		}
	})
}

// subtract subtracts two tensors elementwise

// subtract exposes this operation as part of the public API.

// subtract exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) subtract[T](b &Tensor[T]) !&Tensor[T] {
	return a.map_pair_values[T](b, fn [T](a T, b T) T {
		$if T is bool {
			return td[T](a).bool() && !td[T](b).bool()
		} $else $if T is string {
			return a.replace(b, '')
		} $else {
			return a - b
		}
	})
}

// subtract subtracts a scalar to a tensor elementwise

// subtract_scalar exposes this operation as part of the public API.

// subtract_scalar exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) subtract_scalar[T](scalar T) !&Tensor[T] {
	return a.map_values(fn [scalar] [T](x T) T {
		$if T is bool {
			return td[T](x).bool() && !td[T](scalar).bool()
		} $else $if T is string {
			return x.replace(scalar, '')
		} $else {
			return x - scalar
		}
	})
}

// divide divides two tensors elementwise

// divide exposes this operation as part of the public API.

// divide exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) divide[T](b &Tensor[T]) !&Tensor[T] {
	return a.map_pair_values[T](b, fn [T](a T, b T) T {
		$if T is bool || T is string {
			panic(@FN + ' is not supported for type ${typeof(a).name}')
		} $else {
			return a / b
		}
	})
}

// divide divides a scalar to a tensor elementwise

// divide_scalar exposes this operation as part of the public API.

// divide_scalar exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) divide_scalar[T](scalar T) !&Tensor[T] {
	return a.map_values(fn [scalar] [T](x T) T {
		$if T is bool || T is string {
			panic(@FN + ' is not supported for type ${typeof(x).name}')
		} $else {
			return x / scalar
		}
	})
}

// multiply multiplies two tensors elementwise

// multiply exposes this operation as part of the public API.

// multiply exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) multiply[T](b &Tensor[T]) !&Tensor[T] {
	return a.map_pair_values[T](b, fn [T](a T, b T) T {
		$if T is bool || T is string {
			panic(@FN + ' is not supported for type ${typeof(a).name}')
		} $else {
			return a * b
		}
	})
}

// multiply multiplies a scalar to a tensor elementwise

// multiply_scalar exposes this operation as part of the public API.

// multiply_scalar exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) multiply_scalar[T](scalar T) !&Tensor[T] {
	return a.map_values(fn [scalar] [T](x T) T {
		$if T is bool || T is string {
			panic(@FN + ' is not supported for type ${typeof(x).name}')
		} $else {
			return x * scalar
		}
	})
}

// + adds two tensors elementwise. It panics when their shapes are not broadcastable;
// use add when shape errors need to be handled explicitly.
@[inline]
pub fn (a &Tensor[T]) + (b &Tensor[T]) &Tensor[T] {
	return a.add(b) or { panic(err) }
}

// - subtracts two tensors elementwise, following broadcasting rules.
// It panics when their shapes are not broadcastable; use subtract when shape
// errors need to be handled explicitly.
@[inline]
pub fn (a &Tensor[T]) - (b &Tensor[T]) &Tensor[T] {
	return a.subtract(b) or { panic(err) }
}

// * multiplies two tensors elementwise, following broadcasting rules.
// It panics when their shapes are not broadcastable; use multiply when shape
// errors need to be handled explicitly.
@[inline]
pub fn (a &Tensor[T]) * (b &Tensor[T]) &Tensor[T] {
	return a.multiply(b) or { panic(err) }
}

// / divides two tensors elementwise, following broadcasting rules.
// It panics when their shapes are not broadcastable; use divide when shape
// errors need to be handled explicitly.
@[inline]
pub fn (a &Tensor[T]) / (b &Tensor[T]) &Tensor[T] {
	return a.divide(b) or { panic(err) }
}
