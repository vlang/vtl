module vtl

// array_split splits an array into multiple sub-arrays.
// Please refer to the split documentation. The only difference between
// these functions is that array_split allows indices_or_sections to be an
// integer that does not equally divide the axis. For an array of length
// l that should be split into n sections, it returns l % n sub-arrays of
// size l//n + 1 and the rest of size l//n.
pub fn (t &Tensor<T>) array_split<T>(ind int, axis int) ?[]&Tensor<T> {
	ntotal := t.shape[axis]
	neach := ntotal / ind
	extras := ntotal % ind
	mut sizes := [0]
	sizes << []int{len: extras, init: neach + 1}
	sizes << []int{len: ind - extras, init: neach}
	mut rt := 0
	for i in 0 .. sizes.len {
		tmp := rt
		rt += sizes[i]
		sizes[i] += tmp
	}
	return t.splitter<T>(axis, ind, sizes)
}

// array_split_expl splits an array into multiple sub-arrays.
// Please refer to the split documentation. The only difference between
// these functions is that array_split allows indices_or_sections to be an
// integer that does not equally divide the axis. For an array of length
// l that should be split into n sections, it returns l % n sub-arrays of
// size l//n + 1 and the rest of size l//n.
pub fn (t &Tensor<T>) array_split_expl<T>(ind []int, axis int) ?[]&Tensor<T> {
	nsections := ind.len + 1
	mut div_points := [0]
	div_points << ind
	div_points << [t.shape[axis]]
	return t.splitter<T>(axis, nsections, div_points)
}

// split splits an array into multiple sub-arrays. The array will be divided into
// N equal arrays along axis. If such a split is not possible,
// panic
pub fn (t &Tensor<T>) split<T>(ind int, axis int) ?[]&Tensor<T> {
	n := t.shape[axis]
	if n % ind != 0 {
		return error('Array split does not result in an equal division')
	}
	return t.array_split<T>(ind, axis)
}

// split_expl splits an array into multiple sub-arrays. The array will be divided into
// The entries of ind indicate where along axis the array is split.
// For example, [2, 3] would, for axis=0, result in:
// ary[:2]
// ary[2:3]
// ary[3:]
pub fn (t &Tensor<T>) split_expl<T>(ind []int, axis int) ?[]&Tensor<T> {
	return t.array_split_expl<T>(ind, axis)
}

// hsplit splits an array into multiple sub-arrays horizontally (column-wise).
// Please refer to the split documentation. hsplit is equivalent to
// split with axis=1, the array is always split along the second axis
// regardless of the array dimension.
pub fn (t &Tensor<T>) hsplit<T>(ind int) ?[]&Tensor<T> {
	return match t.rank() {
		1 { t.split<T>(ind, 0)? }
		else { t.split<T>(ind, 1)? }
	}
}

// hsplit_expl splits an array into multiple sub-arrays horizontally (column-wise)
// Please refer to the split documentation. hsplit is equivalent to
// split with axis=1, the array is always split along the second axis
// regardless of the array dimension.
pub fn (t &Tensor<T>) hsplit_expl<T>(ind []int) ?[]&Tensor<T> {
	return match t.rank() {
		1 { t.split_expl<T>(ind, 0)? }
		else { t.split_expl<T>(ind, 1)? }
	}
}

// vsplit splits an array into multiple sub-arrays vertically (row-wise).
// Please refer to the split documentation. vsplit is equivalent to
// split with axis=0 (default), the array is always split along the
// first axis regardless of the array dimension.
pub fn (t &Tensor<T>) vsplit<T>(ind int) ?[]&Tensor<T> {
	if t.rank() < 2 {
		return error('vsplit only works on tensors of >= 2 dimensions')
	}
	return t.split<T>(ind, 0)
}

// vsplit_expl splits an array into multiple sub-arrays vertically (row-wise).
// Please refer to the split documentation. vsplit is equivalent to
// split with axis=0 (default), the array is always split along the
// first axis regardless of the array dimension.
pub fn (t &Tensor<T>) vsplit_expl<T>(ind []int) ?[]&Tensor<T> {
	if t.rank() < 2 {
		return error('vsplit only works on tensors of >= 2 dimensions')
	}
	return t.split_expl<T>(ind, 0)
}

// dsplit splits array into multiple sub-arrays along the 3rd axis (depth).
// Please refer to the split documentation. dsplit is equivalent to
// split with axis=2, the array is always split along the third axis
// provided the array dimension is greater than or equal to 3.
pub fn (t &Tensor<T>) dsplit<T>(ind int) ?[]&Tensor<T> {
	if t.rank() < 3 {
		return error('dsplit only works on arrays of 3 or more dimensions')
	}
	return t.split<T>(ind, 2)
}

// dsplit_expl splits array into multiple sub-arrays along the 3rd axis (depth).
// Please refer to the split documentation. dsplit is equivalent to
// split with axis=2, the array is always split along the third axis
// provided the array dimension is greater than or equal to 3.
pub fn (t &Tensor<T>) dsplit_expl<T>(ind []int) ?[]&Tensor<T> {
	if t.rank() < 3 {
		return error('dsplit only works on arrays of 3 or more dimensions')
	}
	return t.split_expl<T>(ind, 2)
}

// splitter implements a generic splitting function that contains the underlying functionality
// for all split operations
fn (t &Tensor<T>) splitter<T>(axis int, n int, div_points []int) ?[]&Tensor<T> {
	mut subary := []&Tensor<T>{}
	sary := t.swapaxes(axis, 0)?
	for i in 0 .. n {
		st := div_points[i]
		en := div_points[i + 1]
		subary << sary.slice_hilo([st], [en])?.swapaxes(axis, 0)?
	}
	return subary
}
