module num

// Return a new array of given shape and type,
// without initializing entries.
pub fn empty(shape []int) NdArray {
	return allocate_cpu(shape, 'C')
}

// Return a new array with the same shape and type as a
// given array.
pub fn empty_like(t NdArray) NdArray {
	return empty(t.shape)
}

// Return a 2-D array with ones on the diagonal and zeros elsewhere.
pub fn eye(m, n, k int) NdArray {
	ret := zeros([m, n])
	mut iter := ret.iter()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == j - k {
				unsafe {
					*iter.ptr = f64(1.0)
				}
			}
			iter.next()
		}
	}
	return ret
}

// Return the identity array.
//
// The identity array is a square array with ones on the main diagonal.
pub fn identity(n int) NdArray {
	return eye(n, n, 0)
}

// Return a new array of given shape and type, filled with fill_value.
pub fn full(shape []int, val f64) NdArray {
	ret := empty(shape)
	for iter := ret.iter(); !iter.done; iter.next() {
		unsafe {
			*iter.ptr = val
		}
	}
	return ret
}

// Return a full array with the same shape and type as a given array.
pub fn full_like(t NdArray, val f64) NdArray {
	return full(t.shape, val)
}

// Return a new array of given shape and type, filled with zeros.
pub fn zeros(shape []int) NdArray {
	return full(shape, 0.0)
}

// Return an array of zeros with the same shape and type as a given array.
pub fn zeros_like(t NdArray) NdArray {
	return zeros(t.shape)
}

// Return a new array of given shape and type, filled with ones.
pub fn ones(shape []int) NdArray {
	return full(shape, 1.0)
}

// Return an array of ones with the same shape and type as a given array.
pub fn ones_like(t NdArray) NdArray {
	return full(t.shape, 1.0)
}

// from_f64 takes a one dimensional array of floating point values
// and coerces it into an arbitrary shaped ndarray if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_f64(a []f64, shape []int) NdArray {
	data := a.clone()
	size := shape_size(shape)
	if size != a.len {
		panic('Bad shape for array, shape [$a.len] cannot fit into shape $shape')
	}
	return NdArray{
		storage: cpu_from_array(data)
		size: size
		ndims: shape.len
		strides: cstrides(shape)
		shape: shape
		flags: default_flags('C', shape.len)
	}
}

// from_f64_1d takes a one dimensional array of floating point values
// and returns a one dimensional ndarray if possible.
pub fn from_f64_1d(a []f64) NdArray {
	return from_f64(a, [a.len])
}

// from_f64_2d takes a two dimensional array of floating point values
// and returns a two-dimensional ndarray if possible
pub fn from_f64_2d(a [][]f64) NdArray {
	ret := allocate_cpu([a.len, a[0].len], 'C')
	mut iter := ret.iter()
	for i := 0; i < a.len; i++ {
		for j := 0; j < a[0].len; j++ {
			unsafe {
				*iter.ptr = a[i][j]
			}
			iter.next()
		}
	}
	return ret
}

// from_f32 takes a one dimensional array of floating point values
// and coerces it into an arbitrary shaped ndarray if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_f32(a []f32, shape []int) NdArray {
	data := a.map(f64(it))
	size := shape_size(shape)
	if size != a.len {
		panic('Bad shape for array, shape [$a.len] cannot fit into shape $shape')
	}
	return NdArray{
		storage: cpu_from_array(data)
		size: size
		ndims: shape.len
		strides: cstrides(shape)
		shape: shape
		flags: default_flags('C', shape.len)
	}
}

// from_f32_1d takes a one dimensional array of floating point values
// and returns a one dimensional ndarray if possible.
pub fn from_f32_1d(a []f32) NdArray {
	ret := a.map(f64(it))
	return NdArray{
		storage: cpu_from_array(ret)
		size: a.len
		ndims: 1
		flags: default_flags('C', 1)
		strides: [1]
		shape: [a.len]
	}
}

// from_f32_2d takes a two dimensional array of floating point values
// and returns a two-dimensional ndarray if possible
pub fn from_f32_2d(a [][]f32) NdArray {
	ret := allocate_cpu([a.len, a[0].len], 'C')
	mut iter := ret.iter()
	for i := 0; i < a.len; i++ {
		for j := 0; j < a[0].len; j++ {
			unsafe {
				*iter.ptr = f64(a[i][j])
			}
			iter.next()
		}
	}
	return ret
}

// from_int takes a one dimensional array of uinteger values
// and coerces it into an arbitrary shaped ndarray if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_int(a, shape []int) NdArray {
	data := a.map(f64(it))
	size := shape_size(shape)
	if size != a.len {
		panic('Bad shape for array, shape [$a.len] cannot fit into shape $shape')
	}
	return NdArray{
		storage: cpu_from_array(data)
		size: size
		ndims: shape.len
		strides: cstrides(shape)
		shape: shape
		flags: default_flags('C', shape.len)
	}
}

// from_int_1d takes a one dimensional array of integer values
// and returns a one dimensional ndarray if possible.
pub fn from_int_1d(a []int) NdArray {
	ret := a.map(f64(it))
	return NdArray{
		storage: cpu_from_array(ret)
		size: a.len
		ndims: 1
		flags: default_flags('C', 1)
		strides: [1]
		shape: [a.len]
	}
}

// from_int_2d takes a two dimensional array of integer values
// and returns a two-dimensional ndarray if possible
pub fn from_int_2d(a [][]int) NdArray {
	ret := allocate_cpu([a.len, a[0].len], 'C')
	mut iter := ret.iter()
	for i := 0; i < a.len; i++ {
		for j := 0; j < a[0].len; j++ {
			unsafe {
				*iter.ptr = f64(a[i][j])
			}
			iter.next()
		}
	}
	return ret
}
