module num

// broadcastable takes two ndarrays as inputs, and finds the proper
// shape that they both broadcast to in order to find the proper output.
// This may require a change to the shapes of both tensors, so for assignment
// and in-place modification, only the right hand side should be broadcasted.
pub fn broadcastable(arr, other NdArray) []int {
	sz := arr.shape.len
	osz := other.shape.len
	if sz == osz {
		if broadcast_equal(arr.shape, other.shape) {
			return broadcastable_shape(arr.shape, other.shape)
		}
	} else {
		if sz > osz {
			mut othershape := [1].repeat(sz - osz)
			othershape << other.shape
			if broadcast_equal(arr.shape, othershape) {
				return broadcastable_shape(arr.shape, othershape)
			}
		} else {
			mut selfshape := [1].repeat(osz - sz)
			selfshape << arr.shape
			if broadcast_equal(selfshape, other.shape) {
				return broadcastable_shape(selfshape, other.shape)
			}
		}
	}
	panic('Shapes $arr.shape and $other.shape are not broadcastable')
	return [] // needed since every function needs a return statement by default
}

// broadcast2 broadcasts two ndarrays against each other
pub fn broadcast2(a, b NdArray) (NdArray, NdArray) {
	if shape_compare(a.shape, b.shape) {
		return a, b
	} else {
		shape := broadcastable(a, b)
		return broadcast_if(a, shape), broadcast_if(b, shape)
	}
}

// max_dims returns the maximum number of dimensions for any number of ndarrays
fn max_dims(arrs ...NdArray) int {
	mut md := 0
	for i in arrs {
		if i.ndims > md {
			md = i.ndims
		}
	}
	return md
}

// broadcast3 broadcasts three ndarrays against each other
pub fn broadcast3(a, b, c NdArray) (NdArray, NdArray, NdArray) {
	if shape_compare(a.shape, b.shape) && shape_compare(a.shape, c.shape) {
		return a, b, c
	} else {
		nd := max_dims(a, b, c)
		mut shape := [0].repeat(nd)
		t := [a, b, c]
		for i := 0; i < t.len; i++ {
			diff := nd - t[i].ndims
			mut tshape := [1].repeat(diff)
			tshape << t[i].shape
			for j := 0; j < shape.len; j++ {
				if tshape[j] > shape[j] {
					shape[j] = tshape[j]
				}
			}
		}
		return broadcast_if(a, shape), broadcast_if(b, shape), broadcast_if(c, shape)
	}
}

// broadcast4 broadcasts 4 ndarrays against each other
pub fn broadcast4(a, b, c, d NdArray) (NdArray, NdArray, NdArray, NdArray) {
	if shape_compare(a.shape, b.shape) && shape_compare(a.shape, c.shape) && shape_compare(a.shape, d.shape) {
		return a, b, c, d
	} else {
		nd := max_dims(a, b, c, d)
		mut shape := [0].repeat(nd)
		t := [a, b, c, d]
		for i := 0; i < t.len; i++ {
			diff := nd - t[i].ndims
			mut tshape := [1].repeat(diff)
			tshape << t[i].shape
			for j := 0; j < shape.len; j++ {
				if tshape[j] > shape[j] {
					shape[j] = tshape[j]
				}
			}
		}
		return broadcast_if(a, shape), broadcast_if(b, shape), broadcast_if(c, shape), broadcast_if(d,
			shape)
	}
}

// broadcast_if is used internally to only broadcast an array if necessary
fn broadcast_if(a NdArray, shape []int) NdArray {
	if shape_compare(a.shape, shape) {
		return a
	} else {
		return broadcast_to(a, shape)
	}
}

// broadcast_equal checks two shapes, asserting that they can be broadcasted
// according to a couple basic rules: either they are equal or one is equal
// to 1
fn broadcast_equal(a, b []int) bool {
	mut bc := true
	for i, v in a {
		if !(v == b[i] || v == 1 || b[i] == 1) {
			bc = false
		}
	}
	return bc
}

// broadcasts_strides broadcasts the strides of an existing array into a new shape that it is able
// to be broadcast into.  Since a copy is not made, the new strides will
// be heavily dependent on the current memory layout of the existing array
// This will almost never result in a contiguous array
fn broadcast_strides(dest_shape, src_shape, dest_strides, src_strides []int) []int {
	dims := dest_shape.len
	start := dims - src_shape.len
	mut ret := [0].repeat(dims)
	mut i := dims - 1
	for i >= start {
		s := src_shape[i - start]
		if s == 1 {
			ret[i] = 0
		} else if s == dest_shape[i] {
			ret[i] = src_strides[i - start]
		} else {
			panic('Cannot broadcast from $src_shape to $dest_shape')
		}
		i--
	}
	return ret
}

// Returns the final broadcastable shape between two arrays of shapes
// This takes the maximum at each index of the two shapes, and
// the smaller dimension is where the broadcast occurs in
// the derived arrays.
fn broadcastable_shape(a, b []int) []int {
	mut ret := []int{}
	for i, aval in a {
		if aval > b[i] {
			ret << aval
		} else {
			ret << b[i]
		}
	}
	return ret
}

// broadcast_to broadcasts an ndarray to a new shape, panicking if
// the ndarray cannot be viewed as the new shape
pub fn broadcast_to(t NdArray, newshape []int) NdArray {
	defstrides := cstrides(newshape)
	newstrides := broadcast_strides(newshape, t.shape, defstrides, t.strides)
	newflags := no_flags()
	return NdArray{
		storage: t.storage
		shape: newshape
		strides: newstrides
		flags: newflags
		size: shape_size(newshape)
		ndims: newshape.len
	}
}

// as_strided as a highly unsafe method that views an array given
// an arbitrary shape and stride.  The result is not writeable, and
// many elements may share the same memory location.  Be very careful
// using this method, as using it incorrectly can lead to dangerous
// memory access.
pub fn as_strided(t NdArray, newshape, newstrides []int) NdArray {
	return NdArray{
		storage: t.storage
		shape: newshape
		strides: newstrides
		flags: no_flags()
		size: shape_size(newshape)
		ndims: newshape.len
	}
}

// broadcast_arrays takes two input arrays, and if possible, broadcasts
// them into compatible shapes, returning the two modified arrays.
// No copies of data are made, that must be handled later.  If the arrays
// cannot be broadcast against each other, panic.
pub fn broadcast_arrays(a, b NdArray) (NdArray, NdArray) {
	if shape_compare(a.shape, b.shape) {
		return a, b
	}
	newshape := broadcastable(a, b)
	newa := match (shape_compare(a.shape, newshape)) {
		true { a }
		else { broadcast_to(a, newshape) }
	}
	newb := match (shape_compare(b.shape, newshape)) {
		true { b }
		else { broadcast_to(b, newshape) }
	}
	return newa, newb
}

// expand_dims adds an axis to an ndarray in order to support
// broadcasting operations
pub fn expand_dims(a NdArray, axis int) NdArray {
	mut newshape := []int{}
	newaxis := match (axis < 0) {
		true { axis + a.ndims + 1 }
		else { axis }
	}
	newshape << a.shape[..newaxis]
	newshape << 1
	newshape << a.shape[newaxis..]
	return a.reshape(newshape)
}
