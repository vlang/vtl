module vtl

// diag constructs a diagonal array.
// input must be one dimensional and will be placed along the resulting diagonal
pub fn (t &Tensor[T]) diag[T]() !&Tensor[T] {
	if t.rank() != 1 {
		return error('Input array must be 1D. Use diag_flat for higher dimensional arrays')
	}
	mut ret := zeros[T]([t.size, t.size]).diagonal()
	return ret.assign(t)
}

// diag_flat constructs a diagonal array.
// the flattened input is placed along the diagonal of the resulting matrix
pub fn (t &Tensor[T]) diag_flat[T]() !&Tensor[T] {
	mut ret := zeros[T]([t.size, t.size]).diagonal()
	return ret.assign(t.ravel()!)
}

// tril computes the lower triangle of an array.
// returns a copy of an array with elements above the diagonal zeroed
pub fn (t &Tensor[T]) tril[T]() &Tensor[T] {
	mut ret := t.copy(.row_major)
	return ret.tril_inplace_offset[T](0)
}

// tril_inpl computes the lower triangle of an array.
// returns a copy of an array with elements above the kth diagonal zeroed
pub fn (t &Tensor[T]) tril_offset[T](offset int) &Tensor[T] {
	mut ret := t.copy(.row_major)
	return ret.tril_inplace_offset[T](offset)
}

// tril_inpl computes the lower triangle of an array.
// modifies an array inplace with elements above the diagonal zeroed.
pub fn (mut t Tensor[T]) tril_inpl[T]() &Tensor[T] {
	return t.tril_inplace_offset[T](0)
}

// tril_inpl_offset computes the lower triangle of an array.
// modifies an array inplace with elements above the k-th diagonal zeroed.
pub fn (mut t Tensor[T]) tril_inpl_offset[T](offset int) &Tensor[T] {
	return t.tril_inplace_offset[T](offset)
}

// triu computes the Upper triangle of an array.
// returns a copy of an array with elements below the diagonal zeroed
pub fn (t &Tensor[T]) triu[T]() &Tensor[T] {
	mut ret := t.copy(.row_major)
	return ret.triu_inplace_offset[T](0)
}

// triu_offset computes the upper triangle of an array.
// returns a copy of an array with elements below the kth diagonal zeroed
pub fn (t &Tensor[T]) triu_offset[T](offset int) &Tensor[T] {
	mut ret := t.copy(.row_major)
	return ret.triu_inplace_offset[T](offset)
}

// triu_inpl computes the upper triangle of an array.
// modifies an array inplace with elements below the diagonal zeroed.
pub fn (mut t Tensor[T]) triu_inpl[T]() &Tensor[T] {
	return t.triu_inplace_offset[T](0)
}

// triu_inplace_offset computes the uriu_inplace_offset computes the lower triangle of an array.
// modifies an array inplace with elements above the k-th diagonal zeroed.
fn (mut t Tensor[T]) tril_inplace_offset[T](offset int) &Tensor[T] {
	for i in 0 .. t.shape[0] {
		for j in 0 .. t.shape[1] {
			if i < j - offset {
				t.set([i, j], cast[T](0))
			}
		}
	}
	return t
}

// triu_inplace_offset computes the upper triangle of an array.
// modifies an array inplace with elements below the k-th diagonal zeroed.
fn (mut t Tensor[T]) triu_inplace_offset[T](offset int) &Tensor[T] {
	for i in 0 .. t.shape[0] {
		for j in 0 .. t.shape[1] {
			if i > j - offset {
				t.set([i, j], cast[T](0))
			}
		}
	}
	return t
}
