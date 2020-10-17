module num

pub type OneParamFn = fn (x f64) f64

pub type TwoParamsFn = fn (x f64, y f64) f64

pub type ThreeParamsFn = fn (x f64, y f64, z f64) f64

// amap maps a function to a single ndarray
pub fn amap(n NdArray, op OneParamFn) NdArray {
	ret := allocate_cpu(n.shape, 'C')
	for iter := ret.iter2(n); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_b)
		}
	}
	return ret
}

// apply applies a function in place on a single ndarray
pub fn apply(n NdArray, op OneParamFn) {
	for iter := n.iter(); !iter.done; iter.next() {
		unsafe {
			*iter.ptr = op(*iter.ptr)
		}
	}
}

// map_scalar maps a function with a scalar input to an ndarray
pub fn map_scalar(a NdArray, b f64, op TwoParamsFn) NdArray {
	ret := allocate_cpu(a.shape, 'C')
	for iter := ret.iter2(a); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_b, b)
		}
	}
	return ret
}

// map_scalar_lhs maps a function to a scalar and an ndarray
pub fn map_scalar_lhs(a f64, b NdArray, op TwoParamsFn) NdArray {
	ret := allocate_cpu(b.shape, 'C')
	for iter := ret.iter2(b); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(a, *iter.ptr_b)
		}
	}
	return ret
}

// map2 maps a function along two ndarrays
pub fn map2(a NdArray, b NdArray, op TwoParamsFn) NdArray {
	ab, bb := broadcast2(a, b)
	ret := allocate_cpu(ab.shape, 'C')
	for iter := ret.iter3(ab, bb); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_b, *iter.ptr_c)
		}
	}
	return ret
}

// apply2 applies a function to two ndarrays, storing the result in
// the first ndarray
pub fn apply2(a NdArray, b NdArray, op TwoParamsFn) {
	bb := broadcast_if(b, a.shape)
	for iter := a.iter2(bb); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_a, *iter.ptr_b)
		}
	}
}

// map3 maps a function along three ndarrays
pub fn map3(a NdArray, b NdArray, c NdArray, op ThreeParamsFn) NdArray {
	ab, bb, cb := broadcast3(a, b, c)
	ret := allocate_cpu(ab.shape, 'C')
	for iter := ret.iter4(ab, bb, cb); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_b, *iter.ptr_c, *iter.ptr_d)
		}
	}
	return ret
}

// apply3 applies a function to three ndarrays, storing the result in the
// first ndarray
pub fn apply3(a NdArray, b NdArray, c NdArray, op ThreeParamsFn) {
	bb := broadcast_if(b, a.shape)
	cb := broadcast_if(c, a.shape)
	for iter := a.iter3(bb, cb); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = op(*iter.ptr_a, *iter.ptr_b, *iter.ptr_c)
		}
	}
}
