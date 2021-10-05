module vtl

// diag constructs a diagonal array.
// input must be one dimensional and will be placed along the resulting diagonal
pub fn diag<T>(t &Tensor<T>) &Tensor<T> {
	if t.rank() != 1 {
		panic('Input array must be 1D. Use diag_flat for higher dimensional arrays')
	}
	mut ret := zeros<T>([t.size, t.size]).diagonal()
	ret.assign(t)
	return ret
}

// diag_flat constructs a diagonal array.
// the flattened input is placed along the diagonal of the resulting matrix
pub fn diag_flat<T>(t &Tensor<T>) &Tensor<T> {
	mut ret := zeros<T>([t.size, t.size]).diagonal()
	ret.assign(t.ravel())
	return ret
}

// tril computes the lower triangle of an array.
// returns a copy of an array with elements above the diagonal zeroed
pub fn tril<T>(t &Tensor<T>) &Tensor<T> {
	mut ret := t.copy(.row_major)
	tril_inplace_offset<T>(mut ret, 0)
	return ret
}

// tril_inpl computes the lower triangle of an array.
// returns a copy of an array with elements above the kth diagonal zeroed
pub fn tril_offset<T>(t &Tensor<T>, offset int) &Tensor<T> {
	mut ret := t.copy(.row_major)
	tril_inplace_offset<T>(mut ret, offset)
	return ret
}

// tril_inpl computes the lower triangle of an array.
// modifies an array inplace with elements above the diagonal zeroed.
pub fn tril_inpl<T>(mut t Tensor<T>) {
	tril_inplace_offset<T>(mut t, 0)
}

// tril_inpl_offset computes the lower triangle of an array.
// modifies an array inplace with elements above the k-th diagonal zeroed.
pub fn tril_inpl_offset<T>(mut t Tensor<T>, offset int) {
	tril_inplace_offset<T>(mut t, offset)
}

// triu computes the Upper triangle of an array.
// returns a copy of an array with elements below the diagonal zeroed
pub fn triu<T>(t &Tensor<T>) &Tensor<T> {
	mut ret := t.copy(.row_major)
	triu_inplace_offset<T>(mut ret, 0)
	return ret
}

// triu_offset computes the upper triangle of an array.
// returns a copy of an array with elements below the kth diagonal zeroed
pub fn triu_offset<T>(t &Tensor<T>, offset int) &Tensor<T> {
	mut ret := t.copy(.row_major)
	triu_inplace_offset<T>(mut ret, offset)
	return ret
}

// triu_inpl computes the upper triangle of an array.
// modifies an array inplace with elements below the diagonal zeroed.
pub fn triu_inpl<T>(mut t Tensor<T>) {
	triu_inplace_offset<T>(mut t, 0)
}

// triu_inplace_offset computes the uriu_inplace_offset computes the lower triangle of an array.
// modifies an array inplace with elements above the k-th diagonal zeroed.
fn tril_inplace_offset<T>(mut t Tensor<T>, offset int) {
	mut i := 0
	for i < t.shape[0] {
		mut j := 0
		for j < t.shape[1] {
			if i < j - offset {
				val := 0
				t.set([i, j], &val)
			}
			j++
		}
		i++
	}
}

// triu_inplace_offset computes the upper triangle of an array.
// modifies an array inplace with elements below the k-th diagonal zeroed.
fn triu_inplace_offset<T>(mut t Tensor<T>, offset int) {
	mut i := 0
	for i < t.shape[0] {
		mut j := 0
		for j < t.shape[1] {
			if i > j - offset {
				t.set([i, j], T(0))
			}
			j++
		}
		i++
	}
}
