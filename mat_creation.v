module vtl

// Construct a diagonal array.
// Input must be one dimensional and will be placed along
// the resulting diagonal
pub fn diag<T>(t Tensor) Tensor {
	if t.rank() != 1 {
		panic('Input array must be 1D. Use diag_flat for higher dimensional arrays')
	}
	ret := zeros<T>([a.size, a.size])
	ret.diagonal().assign(t)
	return ret
}

// Construct a diagonal array.
// The flattened input is placed along the diagonal
// of the resulting matrix
pub fn diag_flat<T>(a Tensor) Tensor {
	mut ret := zeros<T>([a.size, a.size])
        ret.diagonal().assign(a.ravel())
	return ret
}

// Lower triangle of an array.
// Modifies an array inplace with elements above the k-th diagonal zeroed.
fn tril_inplace_offset(mut t Tensor, offset int) {
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

// Upper triangle of an array.
// Modifies an array inplace with elements below the k-th diagonal zeroed.
fn triu_inplace_offset(mut t Tensor, offset int) {
	mut i := 0
	for i < t.shape[0] {
		mut j := 0
		for j < t.shape[1] {
			if i > j - offset {
                                val := 0
				t.set([i, j], &val)
			}
			j++
		}
		i++
	}
}

// Lower triangle of an array.
// Returns a copy of an array with elements above the diagonal zeroed
pub fn tril(t Tensor) Tensor {
	mut ret := t.copy(.rowmajor)
	tril_inplace_offset(mut ret, 0)
	return ret
}

// Lower triangle of an array.
// Returns a copy of an array with elements above the kth diagonal zeroed
pub fn tril_offset(t Tensor, offset int) Tensor {
	mut ret := t.copy(.rowmajor)
	tril_inplace_offset(mut ret, offset)
	return ret
}

// Lower triangle of an array.
// Modifies an array inplace with elements above the diagonal zeroed.
pub fn tril_inpl(mut t Tensor) {
	tril_inplace_offset(mut t, 0)
}

// Lower triangle of an array.
// Modifies an array inplace with elements above the k-th diagonal zeroed.
pub fn tril_inpl_offset(mut t Tensor, offset int) {
	tril_inplace_offset(mut t, offset)
}

// Upper triangle of an array.
// Returns a copy of an array with elements below the diagonal zeroed
pub fn triu(t Tensor) Tensor {
	mut ret := t.copy(.rowmajor)
	triu_inplace_offset(mut ret, 0)
	return ret
}

// Upper triangle of an array.
// Returns a copy of an array with elements below the kth diagonal zeroed
pub fn triu_offset(t Tensor, offset int) Tensor {
	mut ret := t.copy(.rowmajor)
	triu_inplace_offset(mut ret, offset)
	return ret
}

// Upper triangle of an array.
// Modifies an array inplace with elements below the diagonal zeroed.
pub fn triu_inpl(mut t Tensor) {
	triu_inplace_offset(mut t, 0)
}
