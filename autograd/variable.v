module autograd

import vtl

// Variable is an abstraction of a vtl.Tensor that tracks
// the operations done to the vtl.Tensor. It also keeps
// track of the gradient of the operation if a Variable
// needs to backpropagate.
// This is the fundamental object used in automatic
// differentiation, as well as the neural network aspects
// of VTL

// Variable defines a public data structure for this module.

// Variable defines a public data structure for this module.
@[heap]
pub struct Variable[T] {
pub mut:
	// The value of the Variable.  This should not be edited outside
	// of Variable operations, as other edits will not be tracked
	// and will lead to incorrect results
	value &vtl.Tensor[T] = unsafe { nil }
	// The graph the variable is associated with.  This is a reference,
	// as a variable does not own its context
	context &Context[T] = unsafe { nil }
	// The gradient of the Variable.  This is set as a reference to
	// the value of a Variable unless `backprop` has been called, in
	// which case all related Variables will have their gradient
	// updated correctly
	grad &vtl.Tensor[T] = unsafe { nil }
	// If set to true, this variable will track its operations,
	// otherwise it will act similar to a vtl.Tensor, only calculating
	// forward operations
	requires_grad bool
	// Phase 2: optional forward-only GPU activation (`&vtl.CudaTensor[f64]` when `-d cuda`).
	gpu_activation voidptr = unsafe { nil }
}

// VariableData defines a public data structure for this module.

// VariableData defines a public data structure for this module.
@[params]
pub struct VariableData {
	requires_grad bool = true
}

// variable
pub fn variable[T](context &Context[T], value &vtl.Tensor[T], data VariableData) &Variable[T] {
	grad := if data.requires_grad { vtl.zeros_like[T](value) } else { value }
	return &Variable[T]{
		context:       context
		value:         value
		grad:          grad
		requires_grad: data.requires_grad
	}
}

// slice exposes this operation as part of the public API.
pub fn (v &Variable[T]) slice(idx ...[]int) !&Variable[T] {
	for selector in idx {
		if selector.len == 3 && selector[2] == 0 {
			return error('Variable.slice: step cannot be zero')
		}
	}
	value := v.value.slice(...idx)!
	mapping := slice_backward_mapping(v.value.shape, idx, value.shape)!
	return variable_slice_result[T](v, value, mapping)!
}

// slice_hilo exposes this operation as part of the public API.
pub fn (v &Variable[T]) slice_hilo(idx1 []int, idx2 []int) !&Variable[T] {
	value := v.value.slice_hilo(idx1, idx2)!
	mapping := slice_hilo_backward_mapping(v.value.shape, idx1, idx2, value.shape)!
	return variable_slice_result[T](v, value, mapping)!
}

// is_grad_needed exposes this operation as part of the public API.
pub fn (v &Variable[T]) is_grad_needed() bool {
	return v.requires_grad && !v.context.no_grad
}

// str exposes this operation as part of the public API.
pub fn (v &Variable[T]) str() string {
	return v.value.str()
}

// backprop Back propagates an operation along a computational graph.
// This operation will destroy the operational graph, populating
// the gradients for all variables that are predecessors of
// the Variable this is called on.
// Even if this is called on the first node in a graph, it will
// destroy all descendents of this variable stored by the
// Context
pub fn (mut v Variable[T]) backprop() ! {
	v.grad = vtl.ones_like[T](v.value)
	for v.context.len() > 0 && v.context.last()!.payload.variable != v {
		node := v.context.pop()!
		$if debug {
			print(node.name)
		}
	}
	for v.context.len() > 0 {
		cur_node := v.context.pop()!
		$if debug {
			print(cur_node.name)
		}
		diff_ptrs := cur_node.backward(cur_node.gate, voidptr(cur_node.payload))!
		for i, diff_ptr in diff_ptrs {
			diff := unsafe { &vtl.Tensor[T](diff_ptr) }
			mut parent_i := cur_node.parents[i]
			if parent_i.requires_grad {
				parent_i.grad = accumulate_gradient[T](parent_i.grad, diff)!
			}
		}
	}
}

// accumulate_gradient adds one backward contribution to a variable gradient.
// Matching tensor shapes can be accumulated into the existing gradient buffer,
// avoiding a new tensor allocation for each parent edge in the graph.
@[direct_array_access]
fn accumulate_gradient[T](gradient &vtl.Tensor[T], contribution &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if gradient.shape == contribution.shape {
		mut result := gradient
		if result.is_row_major_contiguous() && contribution.is_row_major_contiguous()
			&& result.data.data.len == result.size && contribution.data.data.len == contribution.size {
			for i in 0 .. result.size {
				result.data.data[i] = result.data.data[i] + contribution.data.data[i]
			}
			return result
		}
		result.napply[T]([contribution], fn [T](values []T, _ []int) T {
			return values[0] + values[1]
		})!
		return result
	}
	return gradient.add[T](contribution)
}
