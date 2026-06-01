module vtl

// add adds two tensors elementwise

// add exposes this operation as part of the public API.

// add exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) add[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap([b], fn [T](xs []T, _ []int) T {
		a := xs[0]
		b := xs[1]
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
	return a.map(fn [scalar] [T](x T, _ []int) T {
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
	return a.nmap([b], fn [T](xs []T, _ []int) T {
		a := xs[0]
		b := xs[1]
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
	return a.map(fn [scalar] [T](x T, _ []int) T {
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
	return a.nmap([b], fn [T](xs []T, _ []int) T {
		a := xs[0]
		b := xs[1]
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
	return a.map(fn [scalar] [T](x T, _ []int) T {
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
	return a.nmap([b], fn [T](xs []T, _ []int) T {
		a := xs[0]
		b := xs[1]
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
	return a.map(fn [scalar] [T](x T, _ []int) T {
		$if T is bool || T is string {
			panic(@FN + ' is not supported for type ${typeof(x).name}')
		} $else {
			return x * scalar
		}
	})
}
