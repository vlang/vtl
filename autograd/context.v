module autograd

import vtl

// A Context keeps track of the computational graph for
// a number of operations.  Variables that interact with each
// other must belong to the same context, or state will be
// lost while tracking operations done.
// The generic type of a context is always going to be a specific
// class of `NdArray`, to allow easy creation of gradients on the
// fly.  Unlike standard `NdArray` operations, a `Context` cannot
// shift it's generic type, and operations resulting in a different
// data type will raise.
[heap]
pub struct Context<T> {
pub mut:
	// A list of all variables present in an operation.
	// This list can contain duplicates
	nodes []&Node<T>
	// If no_grad is set to true, operations will not
	// be cached, and backpropogation will not be possible
	no_grad bool
}

// Contexts can only be initialized as empty, and
// a generic type must be provided
pub fn new_ctx<T>() &Context<T> {
	return &Context<T>{}
}

pub fn (ctx &Context<T>) len() int {
	return ctx.nodes.len
}

pub fn (mut ctx Context<T>) push(node &Node<T>) {
	ctx.nodes << node
}

pub fn (ctx &Context<T>) last() &Node<T> {
	return ctx.nodes.last()
}

pub fn (mut ctx Context<T>) pop() &Node<T> {
	return ctx.nodes.pop()
}

pub struct ContextVariableData<T> {
pub:
	value         &vtl.Tensor<T>
	requires_grad bool = true
}

pub fn (ctx &Context<T>) variable<T>(data ContextVariableData<T>) &Variable<T> {
	return new_variable<T>(VariableData<T>{
		context: ctx
		value: data.value
		requires_grad: data.requires_grad
	})
}

pub fn (ctx &Context<T>) str() string {
	mut str := ''
	for i, node in ctx.nodes {
		if node.parents.len <= 1 {
			str = '$str${node.parents[0].value.shape}'
		} else {
			str = '${str}('
			for pi, parent in node.parents {
				if pi != 0 {
					str = '$str, '
				}
				str = '$str$parent.value.shape'
			}
			str = '$str)'
		}
		str = '$str$node.payload.variable.value.shape'
		if i != ctx.nodes.len - 1 {
			str = '$str\n'
		}
	}
	return str
}
