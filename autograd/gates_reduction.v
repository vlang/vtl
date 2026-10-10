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
	r0 := reduction_gradient_broadcast[T](gradient, g.shape, g.axis)!
	return [r0]
}

fn reduction_gradient_broadcast[T](gradient &vtl.Tensor[T], shape []int, axis int) !&vtl.Tensor[T] {
	if shape.len == 0 {
		return vtl.from_array([gradient.get_nth(0)], []int{})
	}
	mut expanded_gradient := gradient
	if axis >= 0 && gradient.rank() == shape.len - 1 {
		expanded_shape := shape.clone()
		expanded_shape[axis] = 1
		expanded_gradient = gradient.reshape[T](expanded_shape)!
	}
	return expanded_gradient.broadcast_to[T](shape)
}

fn sum_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&SumGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SumGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Sum', voidptr(g), sum_gate_backward_dispatch[T], result, [a])!
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
	broadcasted := reduction_gradient_broadcast[T](gradient, g.shape, g.axis)!
	scale := vtl.cast[T](g.num_elems)
	r0 := broadcasted.divide_scalar[T](scale)!
	return [r0]
}

fn mean_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&MeanGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &MeanGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Mean', voidptr(g), mean_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('MeanGate: a must be a Variable')
		}
	}
}

// CumsumGate reverses the cumulative-sum operation by accumulating output
// gradients from the end of each axis slice toward its beginning.
pub struct CumsumGate[T] {
pub:
	shape []int
	axis  int
}

// cumsum_gate creates a cumulative-sum gradient gate.
pub fn cumsum_gate[T](shape []int, axis int) &CumsumGate[T] {
	return &CumsumGate[T]{
		shape: shape
		axis:  axis
	}
}

// backward computes the reverse cumulative sum of the output gradient.
pub fn (g &CumsumGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	mut result := vtl.zeros[T](g.shape, vtl.TensorData{})
	if result.size == 0 {
		return [result]
	}
	rank := g.shape.len
	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * g.shape[i + 1]
	}
	axis_stride := strides[g.axis]
	n_axis := g.shape[g.axis]
	gradient := payload.variable.grad
	mut outer_idx := []int{len: rank}
	for {
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		mut acc := vtl.cast[T](0)
		for j := n_axis - 1; j >= 0; j-- {
			lin := base_lin + j * axis_stride
			acc += gradient.get_nth(lin)
			result.set_nth(lin, acc)
		}
		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == g.axis {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < g.shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return [result]
}

fn cumsum_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&CumsumGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the cumulative-sum backward rule on its input variable.
pub fn (g &CumsumGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Cumsum', voidptr(g), cumsum_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('CumsumGate: a must be a Variable')
		}
	}
}

// CumprodGate stores the input values needed to differentiate cumulative
// products, including slices that contain zero values.
pub struct CumprodGate[T] {
pub:
	input &vtl.Tensor[T] = unsafe { nil }
	axis  int
}

// cumprod_gate creates a cumulative-product gradient gate.
pub fn cumprod_gate[T](input &vtl.Tensor[T], axis int) &CumprodGate[T] {
	return &CumprodGate[T]{
		input: input
		axis:  axis
	}
}

// backward computes product gradients without division, so zero inputs are
// handled correctly.
pub fn (g &CumprodGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	shape := g.input.shape
	mut result := vtl.zeros[T](shape, vtl.TensorData{})
	if result.size == 0 {
		return [result]
	}
	rank := shape.len
	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}
	axis_stride := strides[g.axis]
	n_axis := shape[g.axis]
	gradient := payload.variable.grad
	mut prefix_products := []T{len: n_axis}
	mut outer_idx := []int{len: rank}
	for {
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		mut prefix_product := vtl.cast[T](1)
		for j := 0; j < n_axis; j++ {
			lin := base_lin + j * axis_stride
			prefix_products[j] = prefix_product
			prefix_product *= g.input.get_nth(lin)
		}
		mut suffix_gradient := vtl.cast[T](0)
		for j := n_axis - 1; j >= 0; j-- {
			lin := base_lin + j * axis_stride
			if j + 1 < n_axis {
				next_lin := lin + axis_stride
				suffix_gradient = gradient.get_nth(lin) + g.input.get_nth(next_lin) * suffix_gradient
			} else {
				suffix_gradient = gradient.get_nth(lin)
			}
			result.set_nth(lin, prefix_products[j] * suffix_gradient)
		}
		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == g.axis {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return [result]
}

fn cumprod_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&CumprodGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the cumulative-product backward rule on its input variable.
pub fn (g &CumprodGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Cumprod', voidptr(g), cumprod_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('CumprodGate: a must be a Variable')
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

fn reshape_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&ReshapeGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &ReshapeGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Reshape', voidptr(g), reshape_gate_backward_dispatch[T], result, [
				a,
			])!
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

fn transpose_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&TransposeGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &TransposeGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Transpose', voidptr(g), transpose_gate_backward_dispatch[T], result, [
				a,
			])!
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

fn concat_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&ConcatGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
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
	register[T]('Concat', voidptr(g), concat_gate_backward_dispatch[T], result, vars)!
}
