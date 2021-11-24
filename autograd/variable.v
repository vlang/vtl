module autograd

import vtl
import vtl.la

// Variable is an abstraction of a vtl.Tensor that tracks
// the operations done to the vtl.Tensor. It also keeps
// track of the gradient of the operation if a Variable
// needs to backpropogate.
// This is the fundamental object used in automatic
// differentiation, as well as the neural network aspects
// of VTL
[heap]
pub struct Variable<T> {
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
pub fn (mut v Variable<T>) backprop() {
	grad := vtl.ones_like<T>(v.value)
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
		// diffs := cur_node.gate.backward(cur_node.payload)
		// for i, diff in diffs {
		// 	mut parent_i := cur_node.parents[i]
		// 	if parent_i.requires_grad {
		// 		parent_i.grad = vtl.add<T>(parent_i.grad, diff)
		// 	}
		// }
	}
}

// add Adds two variables together.
pub fn (v &Variable<T>) add<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.add<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_add_gate<T>()
                gate.cache(result, v, other)
        }

        return result
}

// substract Subtracts two variables.
pub fn (v &Variable<T>) substract<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.substract<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_substract_gate<T>()
                gate.cache(result, v, other)
        }

        return result
}

// multiply Multiplies two variables.
pub fn (v &Variable<T>) multiply<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.multiply<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_multiply_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// divide Divides two variables.
pub fn (v &Variable<T>) divide<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.divide<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_divide_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// pow raises a variable to a power.
pub fn (v &Variable<T>) pow<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.pow<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_pow_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// exp Exponentiates a variable.
pub fn (v &Variable<T>) exp<T>() &Variable<T> {
        result := v.context.variable<T>(vtl.exp<T>(v.value))

        if v.requires_grad {
                gate := new_exp_gate<T>(v)
                gate.cache(result, v)
        }

        return result
}

// matmul Multiplies two matrices.
pub fn (v &Variable<T>) matmul<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(la.matmul<T>(v.value, other.value))

        if v.requires_grad || other.requires_grad {
                gate := new_matmul_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// sin Sine of a variable.
pub fn (v &Variable<T>) sin<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.sin<T>(v.value))

        if v.requires_grad || other.requires_grad {
                gate := new_sin_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// cos Cosine of a variable.
pub fn (v &Variable<T>) cos<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.cos<T>(v.value))

        if v.requires_grad || other.requires_grad {
                gate := new_cos_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}

// tan Tan of a variable.
pub fn (v &Variable<T>) tan<T>(other &Variable<T>) &Variable<T> {
        result := v.context.variable<T>(vtl.tan<T>(v.value))

        if v.requires_grad || other.requires_grad {
                gate := new_tan_gate<T>(v, other)
                gate.cache(result, v, other)
        }

        return result
}
