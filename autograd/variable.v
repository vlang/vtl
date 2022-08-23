module autograd

import vtl

// Variable is an abstraction of a vtl.Tensor that tracks
// the operations done to the vtl.Tensor. It also keeps
// track of the gradient of the operation if a Variable
// needs to backpropogate.
// This is the fundamental object used in automatic
// differentiation, as well as the neural network aspects
// of VTL
[heap]
pub struct Variable<T> {
pub:
	// The value of the Variable.  This should not be edited outside
	// of Variable operations, as other edits will not be tracked
	// and will lead to incorrect results
	value &vtl.Tensor<T>
pub mut:
	// The graph the variable is associated with.  This is a reference,
	// as a variable does not own its context
	context &Context<T>
	// The gradient of the Variable.  This is set as a reference to
	// the value of a Variable unless `backprop` has been called, in
	// which case all related Variables will have their gradient
	// updated correctly
	grad &vtl.Tensor<T>
	// If set to true, this variable will track its operations,
	// otherwise it will act similar to a vtl.Tensor, only calculating
	// forward operations
	requires_grad bool
}

[params]
pub struct VariableData {
	requires_grad bool = true
}

// new_variable
pub fn new_variable<T>(context &Context<T>, value &vtl.Tensor<T>, data VariableData) &Variable<T> {
	grad := if data.requires_grad { vtl.zeros_like<T>(value) } else { value }
	return &Variable<T>{
		context: context
		value: value
		grad: grad
		requires_grad: data.requires_grad
	}
}

pub fn (v &Variable<T>) is_grad_needed() bool {
	return v.requires_grad && !v.context.no_grad
}

pub fn (v &Variable<T>) str() string {
	return v.value.str()
}

// backprop Back propogates an operation along a computational graph.
// This operation will destroy the operational graph, populating
// the gradients for all variables that are predecessors of
// the Variable this is called on.
// Even if this is called on the first node in a graph, it will
// destroy all descendents of this variable stored by the
// Context
pub fn (mut v Variable<T>) backprop<T>() ? {
	v.grad = vtl.ones_like<T>(v.value)
	for v.context.len() > 0 && v.context.last().payload.variable != v {
		node := v.context.pop()
		$if debug {
			print(node.name)
		}
	}
	for v.context.len() > 0 {
		cur_node := v.context.pop()
		$if debug {
			print(cur_node.name)
		}
		diffs := gate_backward<T>(cur_node.gate, cur_node.payload)?
		for i, diff in diffs {
			mut parent_i := cur_node.parents[i]
			if parent_i.requires_grad {
				parent_i.grad = vtl.add<T>(parent_i.grad, diff)?
			}
		}
	}
}
