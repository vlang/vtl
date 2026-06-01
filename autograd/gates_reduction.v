module autograd

import vtl

// SumGate implements sum reduction.
// backward: grad broadcast back to input shape
pub struct SumGate[T] {
pub:
	shape []int
	axis  int
}

// sum_gate exposes this operation as part of the public API.
pub fn sum_gate[T](shape []int, axis int) &SumGate[T] {
	return &SumGate[T]{
		shape: shape
		axis:  axis
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SumGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// Broadcast gradient back to original shape
	r0 := gradient.broadcast_to[T](g.shape)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &SumGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Sum', g, result, [a])!
		}
		else {
			return error('SumGate: a must be a Variable')
		}
	}
}

// MeanGate implements mean reduction.
// backward: grad broadcast to input shape / num_elements
pub struct MeanGate[T] {
pub:
	shape     []int
	axis      int
	num_elems int
}

// mean_gate exposes this operation as part of the public API.
pub fn mean_gate[T](shape []int, axis int, num_elems int) &MeanGate[T] {
	return &MeanGate[T]{
		shape:     shape
		axis:      axis
		num_elems: num_elems
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &MeanGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	broadcasted := gradient.broadcast_to[T](g.shape)!
	scale := vtl.cast[T](g.num_elems)
	r0 := broadcasted.divide_scalar[T](scale)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &MeanGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Mean', g, result, [a])!
		}
		else {
			return error('MeanGate: a must be a Variable')
		}
	}
}

// ReshapeGate stores the original shape for backward pass.
// backward: grad reshaped back to original shape
pub struct ReshapeGate[T] {
pub:
	orig_shape []int
}

// reshape_gate exposes this operation as part of the public API.
pub fn reshape_gate[T](orig_shape []int) &ReshapeGate[T] {
	return &ReshapeGate[T]{
		orig_shape: orig_shape
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &ReshapeGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.reshape[T](g.orig_shape)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &ReshapeGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Reshape', g, result, [a])!
		}
		else {
			return error('ReshapeGate: a must be a Variable')
		}
	}
}

// TransposeGate stores the permutation for backward.
// backward: grad transposed back with inverse permutation
pub struct TransposeGate[T] {
pub:
	perm  []int
	iperm []int
}

// transpose_gate exposes this operation as part of the public API.
pub fn transpose_gate[T](perm []int) &TransposeGate[T] {
	// Compute inverse permutation
	mut iperm := []int{len: perm.len}
	for i, p in perm {
		iperm[p] = i
	}
	return &TransposeGate[T]{
		perm:  perm
		iperm: iperm
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &TransposeGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// Transpose with inverse permutation to get original gradient
	r0 := gradient.transpose(g.iperm)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &TransposeGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Transpose', g, result, [a])!
		}
		else {
			return error('TransposeGate: a must be a Variable')
		}
	}
}

// ConcatGate concatenates multiple tensors along an axis.
// backward: split gradient back into original inputs
pub struct ConcatGate[T] {
pub:
	axis   int
	splits []int // size of each input along the concat axis
}

// concat_gate exposes this operation as part of the public API.
pub fn concat_gate[T](axis int, splits []int) &ConcatGate[T] {
	return &ConcatGate[T]{
		axis:   axis
		splits: splits
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &ConcatGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// Split gradient back into len(splits) tensors
	mut results := []&vtl.Tensor[T]{}
	mut offset := 0
	for split in g.splits {
		mut lo := []int{len: gradient.rank()}
		mut hi := []int{len: gradient.rank()}
		for i := 0; i < gradient.rank(); i++ {
			if i == g.axis {
				lo[i] = offset
				hi[i] = offset + split
			} else {
				lo[i] = 0
				hi[i] = gradient.shape[i]
			}
		}
		split_t := gradient.slice_hilo(lo, hi)!
		results << split_t
		offset += split
	}
	return results
}

// cache exposes this operation as part of the public API.
pub fn (g &ConcatGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	mut vars := []&Variable[T]{}
	for arg in args {
		match arg {
			Variable[T] {
				vars << arg
			}
			else {}
		}
	}
	register[T]('Concat', g, result, vars)!
}
