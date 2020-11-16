module vtl

fn strides_from_shape(shape []int, memory MemoryFormat) []int {
	mut accum := 1
	mut result := []int{len: shape.len}
	if memory == .rowmajor {
		for i := shape.len - 1; i >= 0; i-- {
			result[i] = accum
			accum *= shape[i]
		}
		return result
	}
	for i in 0 .. shape.len {
		result[i] = accum
		accum *= shape[i]
	}
	return result
}

fn size_from_shape(shape []int) int {
	mut accum := 1
	for i in shape {
		accum *= i
	}
	return accum
}

// assert_shape_off_axis ensures that the shapes of Tensors match
// for concatenation, except along the axis being joined
fn assert_shape_off_axis(ts []Tensor, axis int, shape []int) []int {
	mut retshape := shape.clone()
	for t in ts {
		if t.shape.len != retshape.len {
			panic('All inputs must share the same number of axes')
		}
		mut i := 0
		for i < shape.len {
			if (i != axis) && (t.shape[i] != shape[i]) {
				panic('All inputs must share a shape off axis')
			}
			i++
		}
		retshape[axis] += t.shape[axis]
	}
	return retshape
}

fn assert_shape(shape []int, ts []Tensor) {
	for t in ts {
		if shape != t.shape {
			panic('All shapes must be equal')
		}
	}
}
