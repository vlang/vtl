module vtl

// Split an array into multiple sub-arrays.
// Please refer to the split documentation. The only difference between
// these functions is that array_split allows indices_or_sections to be an
// integer that does not equally divide the axis. For an array of length
// l that should be split into n sections, it returns l % n sub-arrays of
// size l//n + 1 and the rest of size l//n.
pub fn array_split(t Tensor, ind int, axis int) []Tensor {
	ntotal := t.shape[axis]
	neach := ntotal / ind
	extras := ntotal % ind
	mut sizes := [0]
	sizes << []int{len: extras, init: neach + 1}
	sizes << []int{len: ind - extras, init: neach}
	mut rt := 0
	for i := 0; i < sizes.len; i++ {
		tmp := rt
		rt += sizes[i]
		sizes[i] += tmp
	}
	return splitter(t, axis, ind, sizes)
}

// Split an array into multiple sub-arrays.
// Please refer to the split documentation. The only difference between
// these functions is that array_split allows indices_or_sections to be an
// integer that does not equally divide the axis. For an array of length
// l that should be split into n sections, it returns l % n sub-arrays of
// size l//n + 1 and the rest of size l//n.
pub fn array_split_expl(t Tensor, ind []int, axis int) []Tensor {
	nsections := ind.len + 1
	mut div_points := [0]
	div_points << ind
	div_points << [t.shape[axis]]
	return splitter(t, axis, nsections, div_points)
}

// Split an array into multiple sub-arrays. The array will be divided into
// N equal arrays along axis. If such a split is not possible,
// panic
pub fn split(t Tensor, ind int, axis int) []Tensor {
	n := t.shape[axis]
	if n % ind != 0 {
		panic('Array split does not result in an equal division')
	}
	return array_split(t, ind, axis)
}

// Split an array into multiple sub-arrays. The array will be divided into
// The entries of ind indicate where along axis the array is split.
// For example, [2, 3] would, for axis=0, result in:
// ary[:2]
// ary[2:3]
// ary[3:]
pub fn split_expl(t Tensor, ind []int, axis int) []Tensor {
	return array_split_expl(t, ind, axis)
}

// Split an array into multiple sub-arrays horizontally (column-wise).
// Please refer to the split documentation. hsplit is equivalent to
// split with axis=1, the array is always split along the second axis
// regardless of the array dimension.
pub fn hsplit(t Tensor, ind int) []Tensor {
	return match t.rank() {
		1 { split(t, ind, 0) }
		else { split(t, ind, 1) }
	}
}

// Split an array into multiple sub-arrays horizontally (column-wise)
// Please refer to the split documentation. hsplit is equivalent to
// split with axis=1, the array is always split along the second axis
// regardless of the array dimension.
pub fn hsplit_expl(t Tensor, ind []int) []Tensor {
	return match t.rank() {
		1 { split_expl(t, ind, 0) }
		else { split_expl(t, ind, 1) }
	}
}

// Split an array into multiple sub-arrays vertically (row-wise).
// Please refer to the split documentation. vsplit is equivalent to
// split with axis=0 (default), the array is always split along the
// first axis regardless of the array dimension.
pub fn vsplit(t Tensor, ind int) []Tensor {
	if t.rank() < 2 {
		panic('vsplit only works on tensors of >= 2 dimensions')
	}
	return split(t, ind, 0)
}

// Split an array into multiple sub-arrays vertically (row-wise).
// Please refer to the split documentation. vsplit is equivalent to
// split with axis=0 (default), the array is always split along the
// first axis regardless of the array dimension.
pub fn vsplit_expl(t Tensor, ind []int) []Tensor {
	if t.rank() < 2 {
		panic('vsplit only works on tensors of >= 2 dimensions')
	}
	return split_expl(t, ind, 0)
}

// Split array into multiple sub-arrays along the 3rd axis (depth).
// Please refer to the split documentation. dsplit is equivalent to
// split with axis=2, the array is always split along the third axis
// provided the array dimension is greater than or equal to 3.
pub fn dsplit(t Tensor, ind int) []Tensor {
	if t.rank() < 3 {
		panic('dsplit only works on arrays of 3 or more dimensions')
	}
	return split(t, ind, 2)
}

// Split array into multiple sub-arrays along the 3rd axis (depth).
// Please refer to the split documentation. dsplit is equivalent to
// split with axis=2, the array is always split along the third axis
// provided the array dimension is greater than or equal to 3.
pub fn dsplit_expl(t Tensor, ind []int) []Tensor {
	if t.rank() < 3 {
		panic('dsplit only works on arrays of 3 or more dimensions')
	}
	return split_expl(t, ind, 2)
}

// Generic splitting function that contains the underlying functionality
// for all split operations, should not be made public.
fn splitter(t Tensor, axis int, n int, div_points []int) []Tensor {
	mut subary := []Tensor{}
	sary := t.swapaxes(axis, 0)
	for i := 0; i < n; i++ {
		st := div_points[i]
		en := div_points[i + 1]
		subary << sary.slice([st], [en]).swapaxes(axis, 0)
	}
	return subary
}
