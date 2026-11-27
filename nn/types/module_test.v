module types

import vtl
import vtl.autograd

fn test_module_named_parameter_and_buffer_traversal() ! {
	ctx := autograd.ctx[f64]()
	mut root := module[f64]()
	mut child := module[f64]()
	weight := ctx.variable(vtl.from_array([1.0, 2.0], [2])!)
	bias := ctx.variable(vtl.from_array([3.0], [1])!)
	running_mean := vtl.from_array([4.0], [1])!
	running_var := vtl.from_array([5.0], [1])!

	root.register_parameter('weight', weight)!
	root.register_buffer('running_mean', running_mean)!
	child.register_parameter('bias', bias)!
	child.register_buffer('running_var', running_var)!
	root.register_module('encoder', child)!

	named_parameters := root.named_parameters()
	assert named_parameters.map(it.name) == ['weight', 'encoder.bias']
	assert named_parameters[0].value == weight
	assert named_parameters[1].value == bias
	parameters := root.parameters()
	assert parameters.len == 2
	assert parameters[0] == weight
	assert parameters[1] == bias

	named_buffers := root.named_buffers()
	assert named_buffers.map(it.name) == ['running_mean', 'encoder.running_var']
	assert named_buffers[0].value == running_mean
	assert named_buffers[1].value == running_var
	buffers := root.buffers()
	assert buffers.len == 2
	assert buffers[0] == running_mean
	assert buffers[1] == running_var
}

fn test_module_registration_rejects_duplicate_names_across_kinds() ! {
	ctx := autograd.ctx[f64]()
	mut state := module[f64]()
	parameter := ctx.variable(vtl.from_array([1.0], [1])!)
	buffer := vtl.from_array([2.0], [1])!
	child := module[f64]()

	state.register_parameter('state', parameter)!
	state.register_buffer('state', buffer) or {
		assert err.msg().contains('already in use')
		assert state.named_buffers().len == 0
		return
	}
	assert false, 'expected duplicate parameter/buffer name to fail'
}

fn test_module_registration_rejects_invalid_names_and_cycles() ! {
	ctx := autograd.ctx[f64]()
	mut root := module[f64]()
	mut child := module[f64]()
	parameter := ctx.variable(vtl.from_array([1.0], [1])!)

	root.register_parameter('', parameter) or {
		assert err.msg().contains('non-empty')
	}
	root.register_parameter('nested.value', parameter) or {
		assert err.msg().contains('cannot contain')
	}

	child.register_module('parent', root)!
	root.register_module('child', child) or {
		assert err.msg().contains('cycle')
		assert root.named_parameters().len == 0
		return
	}
	assert false, 'expected cyclic module registration to fail'
}

fn test_module_registration_rejects_duplicate_child_name() ! {
	mut root := module[f64]()
	child := module[f64]()
	root.register_module('block', child)!
	root.register_module('block', child) or {
		assert err.msg().contains('already in use')
		assert root.named_parameters().len == 0
		return
	}
	assert false, 'expected duplicate child name to fail'
}
