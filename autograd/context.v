module autograd

import vtl

// Context keeps track of the computational graph for
// a number of operations. Variables that interact with each
// other must belong to the same context, or state will be
// lost while tracking operations done.
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

pub fn (mut ctx Context<T>) push<T>(node &Node<T>) {
	ctx.nodes << node
}

pub fn (ctx &Context<T>) last<T>() &Node<T> {
	return ctx.nodes.last()
}

pub fn (mut ctx Context<T>) pop<T>() &Node<T> {
	return ctx.nodes.pop()
}

[params]
pub struct ContextVariableData {
	requires_grad bool = true
}

pub fn (ctx &Context<T>) variable<T>(value &vtl.Tensor<T>, data ContextVariableData) &Variable<T> {
	return new_variable<T>(ctx, value, requires_grad: data.requires_grad)
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

pub fn register<T>(name string, gate Gate, result &Variable<T>, parents []&Variable<T>) ? {
	assert parents.len > 0
	if parents.len == 0 {
		return error(@FN + ': it is needed to specify at least one parent')
	}

	payload := new_payload<T>(result)
	node := new_node<T>(gate, parents, payload, name)
	mut ctx := parents[0].context
	ctx.push(node)
}
