module num

// The core iterator for an ndarray, iterates through a flattened
// view of an array, always treating the memory layout as
// c-contigious, so it doesn't matter what the memory layout
// of arrays passed is.
struct NdIter {
pub mut:
	ptr         &f64
	coord       []int
	backstrides []int
	shape       []int
	strides     []int
	dim         int
	done        bool
}

// next increments an nd-iterator by a single value, returning true
// if the iterator has been exhausted and updating the `done` flag.
// This function always increments arrays that are fortran contiguous
// and c-contiguous in the same order, so no adjustments need to be
// made.
// 
// TODO: optimize for c-contiguous arrays that can be iterated over
// in a flat order.
pub fn (mut iter NdIter) next() bool {
	if iter.done {
		return true
	}
	mut coord := iter.coord
	for k := iter.dim; k >= 0; k-- {
		if coord[k] < iter.shape[k] - 1 {
			coord[k]++
			unsafe {
				iter.ptr += iter.strides[k]
			}
			break
		} else {
			if k == 0 {
				iter.done = true
			}
			coord[k] = 0
			unsafe {
				iter.ptr -= iter.backstrides[k]
			}
		}
	}
	return false
}

// iter returns a flat iterator over an ndarray, commonly used
// for reduction operations
pub fn (n NdArray) iter() &NdIter {
	mut backstrides := [0].repeat(n.ndims)
	for i := 0; i < n.ndims; i++ {
		backstrides[i] = n.strides[i] * (n.shape[i] - 1)
	}
	return &NdIter{
		ptr: n.storage.stride_offset(n.shape, n.strides)
		coord: [0].repeat(n.ndims)
		backstrides: backstrides
		shape: n.shape
		strides: n.strides
		dim: n.ndims - 1
		done: false
	}
}

// The core iterator for two ndarrays, iterates through a flattened
// view of two arrays, always treating the memory layout as
// c-contigious, so it doesn't matter what the memory layout
// of arrays passed is.
struct NdIter2 {
pub mut:
	ptr_a         &f64
	ptr_b         &f64
	coord         []int
	backstrides_a []int
	backstrides_b []int
	shape         []int
	strides_a     []int
	strides_b     []int
	dim           int
	done          bool
}

// next increments an nd-iterator by a single value, returning true
// if the iterator has been exhausted and updating the `done` flag.
// This function always increments arrays that are fortran contiguous
// and c-contiguous in the same order, so no adjustments need to be
// made.
// 
// TODO: optimize for c-contiguous arrays that can be iterated over
// in a flat order.
pub fn (mut iter NdIter2) next() bool {
	if iter.done {
		return true
	}
	mut coord := iter.coord
	for k := iter.dim; k >= 0; k-- {
		if coord[k] < iter.shape[k] - 1 {
			coord[k]++
			unsafe {
				iter.ptr_a += iter.strides_a[k]
			}
			unsafe {
				iter.ptr_b += iter.strides_b[k]
			}
			break
		} else {
			if k == 0 {
				iter.done = true
			}
			coord[k] = 0
			unsafe {
				iter.ptr_a -= iter.backstrides_a[k]
			}
			unsafe {
				iter.ptr_b -= iter.backstrides_b[k]
			}
		}
	}
	return false
}

// iter2 returns a flat iterator over two ndarrays, commonly used
// for reduction operations
pub fn (n NdArray) iter2(other NdArray) &NdIter2 {
	mut backstrides_a := [0].repeat(n.ndims)
	mut backstrides_b := [0].repeat(n.ndims)
	for i := 0; i < n.ndims; i++ {
		backstrides_a[i] = n.strides[i] * (n.shape[i] - 1)
		backstrides_b[i] = other.strides[i] * (other.shape[i] - 1)
	}
	return &NdIter2{
		ptr_a: n.storage.stride_offset(n.shape, n.strides)
		ptr_b: other.storage.stride_offset(other.shape, other.strides)
		coord: [0].repeat(n.ndims)
		shape: n.shape
		backstrides_a: backstrides_a
		backstrides_b: backstrides_b
		strides_a: n.strides
		strides_b: other.strides
		dim: n.ndims - 1
		done: false
	}
}

// The core iterator for two ndarrays, iterates through a flattened
// view of two arrays, always treating the memory layout as
// c-contigious, so it doesn't matter what the memory layout
// of arrays passed is.
struct NdIter3 {
pub mut:
	ptr_a         &f64
	ptr_b         &f64
	ptr_c         &f64
	coord         []int
	backstrides_a []int
	backstrides_b []int
	backstrides_c []int
	shape         []int
	strides_a     []int
	strides_b     []int
	strides_c     []int
	dim           int
	done          bool
}

// next increments an nd-iterator by a single value, returning true
// if the iterator has been exhausted and updating the `done` flag.
// This function always increments arrays that are fortran contiguous
// and c-contiguous in the same order, so no adjustments need to be
// made.
// 
// TODO: optimize for c-contiguous arrays that can be iterated over
// in a flat order.
pub fn (mut iter NdIter3) next() bool {
	if iter.done {
		return true
	}
	mut coord := iter.coord
	for k := iter.dim; k >= 0; k-- {
		if coord[k] < iter.shape[k] - 1 {
			coord[k]++
			unsafe {
				iter.ptr_a += iter.strides_a[k]
			}
			unsafe {
				iter.ptr_b += iter.strides_b[k]
			}
			unsafe {
				iter.ptr_c += iter.strides_c[k]
			}
			break
		} else {
			if k == 0 {
				iter.done = true
			}
			coord[k] = 0
			unsafe {
				iter.ptr_a -= iter.backstrides_a[k]
			}
			unsafe {
				iter.ptr_b -= iter.backstrides_b[k]
			}
			unsafe {
				iter.ptr_c -= iter.backstrides_c[k]
			}
		}
	}
	return false
}

// iter2 returns a flat iterator over two ndarrays, commonly used
// for reduction operations
pub fn (n NdArray) iter3(other, other2 NdArray) &NdIter3 {
	mut backstrides_a := [0].repeat(n.ndims)
	mut backstrides_b := [0].repeat(n.ndims)
	mut backstrides_c := [0].repeat(n.ndims)
	for i := 0; i < n.ndims; i++ {
		backstrides_a[i] = n.strides[i] * (n.shape[i] - 1)
		backstrides_b[i] = other.strides[i] * (other.shape[i] - 1)
		backstrides_c[i] = other2.strides[i] * (other2.shape[i] - 1)
	}
	return &NdIter3{
		ptr_a: n.storage.stride_offset(n.shape, n.strides)
		ptr_b: other.storage.stride_offset(other.shape, other.strides)
		ptr_c: other2.storage.stride_offset(other2.shape, other2.strides)
		coord: [0].repeat(n.ndims)
		shape: n.shape
		backstrides_a: backstrides_a
		backstrides_b: backstrides_b
		backstrides_c: backstrides_c
		strides_a: n.strides
		strides_b: other.strides
		strides_c: other2.strides
		dim: n.ndims - 1
		done: false
	}
}

// The core iterator for two ndarrays, iterates through a flattened
// view of two arrays, always treating the memory layout as
// c-contigious, so it doesn't matter what the memory layout
// of arrays passed is.
struct NdIter4 {
pub mut:
	ptr_a         &f64
	ptr_b         &f64
	ptr_c         &f64
	ptr_d         &f64
	coord         []int
	backstrides_a []int
	backstrides_b []int
	backstrides_c []int
	backstrides_d []int
	shape         []int
	strides_a     []int
	strides_b     []int
	strides_c     []int
	strides_d     []int
	dim           int
	done          bool
}

// next increments an nd-iterator by a single value, returning true
// if the iterator has been exhausted and updating the `done` flag.
// This function always increments arrays that are fortran contiguous
// and c-contiguous in the same order, so no adjustments need to be
// made.
// 
// TODO: optimize for c-contiguous arrays that can be iterated over
// in a flat order.
pub fn (mut iter NdIter4) next() bool {
	if iter.done {
		return true
	}
	mut coord := iter.coord
	for k := iter.dim; k >= 0; k-- {
		if coord[k] < iter.shape[k] - 1 {
			coord[k]++
			unsafe {
				iter.ptr_a += iter.strides_a[k]
			}
			unsafe {
				iter.ptr_b += iter.strides_b[k]
			}
			unsafe {
				iter.ptr_c += iter.strides_c[k]
			}
			unsafe {
				iter.ptr_d += iter.strides_d[k]
			}
			break
		} else {
			if k == 0 {
				iter.done = true
			}
			coord[k] = 0
			unsafe {
				iter.ptr_a -= iter.backstrides_a[k]
			}
			unsafe {
				iter.ptr_b -= iter.backstrides_b[k]
			}
			unsafe {
				iter.ptr_c -= iter.backstrides_c[k]
			}
			unsafe {
				iter.ptr_d -= iter.backstrides_d[k]
			}
		}
	}
	return false
}

// iter2 returns a flat iterator over two ndarrays, commonly used
// for reduction operations
pub fn (n NdArray) iter4(other, other2, other3 NdArray) &NdIter4 {
	mut backstrides_a := [0].repeat(n.ndims)
	mut backstrides_b := [0].repeat(n.ndims)
	mut backstrides_c := [0].repeat(n.ndims)
	mut backstrides_d := [0].repeat(n.ndims)
	for i := 0; i < n.ndims; i++ {
		backstrides_a[i] = n.strides[i] * (n.shape[i] - 1)
		backstrides_b[i] = other.strides[i] * (other.shape[i] - 1)
		backstrides_c[i] = other2.strides[i] * (other2.shape[i] - 1)
		backstrides_d[i] = other3.strides[i] * (other3.shape[i] - 1)
	}
	return &NdIter4{
		ptr_a: n.storage.stride_offset(n.shape, n.strides)
		ptr_b: other.storage.stride_offset(other.shape, other.strides)
		ptr_c: other2.storage.stride_offset(other2.shape, other2.strides)
		ptr_d: other3.storage.stride_offset(other3.shape, other3.strides)
		coord: [0].repeat(n.ndims)
		shape: n.shape
		backstrides_a: backstrides_a
		backstrides_b: backstrides_b
		backstrides_c: backstrides_c
		strides_a: n.strides
		strides_b: other.strides
		strides_c: other2.strides
		strides_d: other3.strides
		dim: n.ndims - 1
		done: false
	}
}

// AxesIter is the core iterator for axis-wise operations.
// Stores a copy of an ndarray reduced along a given axis
struct AxesIter {
pub mut:
	ptr     &f64
	shape   []int
	strides []int
	inc     int
	tmp     NdArray
	axis    int
	size    int
}

// next increments the axes iter to store the next reduced
// ndarray along an axis
pub fn (mut iter AxesIter) next() NdArray {
	ret := iter.tmp
	unsafe {
		iter.ptr += iter.inc
	}
	storage := CpuStorage{
		buffer: iter.ptr
	}
	iter.tmp = NdArray{
		storage: storage
		shape: ret.shape
		strides: iter.strides
		flags: ret.flags
		ndims: ret.ndims
		size: ret.size
	}
	return ret
}

// axis returns an iterator over the axis of an ndarray, commonly
// used for reduction operations along an axis.
pub fn (t NdArray) axis(i int) AxesIter {
	mut shape := t.shape.clone()
	shape.delete(i)
	mut strides := t.strides.clone()
	strides.delete(i)
	ptr := t.storage.ptr()
	inc := t.strides[i]
	tmp := NdArray{
		shape: shape
		strides: strides
		storage: t.storage
		size: shape_size(shape)
		ndims: shape.len
		flags: no_flags()
	}
	return AxesIter{
		ptr: ptr
		shape: shape
		strides: strides
		tmp: tmp
		inc: inc
		axis: i
		size: t.shape[i]
	}
}

// axis returns an iterator over the axis of an ndarray, commonly
// used for reduction operations along an axis. This iterator
// keeps the axis dimension as size 1 instead of removing it
pub fn (t NdArray) axis_with_dims(i int) AxesIter {
	mut shape := t.shape.clone()
	shape[i] = 1
	mut strides := t.strides.clone()
	strides[i] = 0
	ptr := t.storage.ptr()
	inc := t.strides[i]
	tmp := NdArray{
		shape: shape
		strides: strides
		storage: t.storage
		size: shape_size(shape)
		ndims: shape.len
		flags: no_flags()
	}
	return AxesIter{
		ptr: ptr
		shape: shape
		strides: strides
		tmp: tmp
		inc: inc
		axis: i
		size: t.shape[i]
	}
}
