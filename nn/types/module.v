module types

import vtl
import vtl.autograd

// NamedParameter associates a custom-module parameter with its registration name.
pub struct NamedParameter[T] {
pub:
	name  string
	value &autograd.Variable[T]
}

// NamedBuffer associates non-parameter state with its registration name.
pub struct NamedBuffer[T] {
pub:
	name  string
	value &vtl.Tensor[T]
}

struct NamedModule[T] {
	name   string
	module &Module[T]
}

// Module stores named parameters, buffers, and child modules for user-defined
// neural-network components. Forward behavior remains the responsibility of
// the custom component; train/eval mode is tracked and propagated to children.
pub struct Module[T] {
pub:
	training bool = true
mut:
	parameters []NamedParameter[T]
	buffers    []NamedBuffer[T]
	children   []NamedModule[T]
}

// module creates an empty custom-module state registry.
pub fn module[T]() &Module[T] {
	return &Module[T]{}
}

// train enables training behavior for this module and all registered children.
pub fn (mut module Module[T]) train() {
	module.set_training(true)
}

// eval enables evaluation behavior for this module and all registered children.
pub fn (mut module Module[T]) eval() {
	module.set_training(false)
}

// set_training updates this module and every registered child recursively.
pub fn (mut module Module[T]) set_training(training bool) {
	module.training = training
	for child in module.children {
		mut child_module := child.module
		child_module.set_training(training)
	}
}

fn validate_module_name(name string) ! {
	if name == '' || name.contains('.') {
		return error('module registration names must be non-empty and cannot contain "."')
	}
}

fn (module &Module[T]) has_name(name string) bool {
	for parameter in module.parameters {
		if parameter.name == name {
			return true
		}
	}
	for buffer in module.buffers {
		if buffer.name == name {
			return true
		}
	}
	for child in module.children {
		if child.name == name {
			return true
		}
	}
	return false
}

// register_parameter adds a trainable variable to this module. Names share one
// namespace with buffers and child modules.
pub fn (mut module Module[T]) register_parameter(name string, value &autograd.Variable[T]) ! {
	validate_module_name(name)!
	if module.has_name(name) {
		return error('module registration name "${name}" is already in use')
	}
	module.parameters << NamedParameter[T]{
		name:  name
		value: value
	}
}

// register_buffer adds non-trainable tensor state to this module. Buffers are
// traversed separately from optimizer parameters.
pub fn (mut module Module[T]) register_buffer(name string, value &vtl.Tensor[T]) ! {
	validate_module_name(name)!
	if module.has_name(name) {
		return error('module registration name "${name}" is already in use')
	}
	module.buffers << NamedBuffer[T]{
		name:  name
		value: value
	}
}

fn module_contains[T](module &Module[T], target voidptr, mut seen []voidptr) bool {
	if voidptr(module) == target {
		return true
	}
	if voidptr(module) in seen {
		return false
	}
	seen << voidptr(module)
	for child in module.children {
		if module_contains[T](child.module, target, mut seen) {
			return true
		}
	}
	return false
}

// register_module attaches a child registry. Cyclic module graphs are rejected
// so recursive state traversal always terminates.
pub fn (mut module Module[T]) register_module(name string, child &Module[T]) ! {
	validate_module_name(name)!
	if module.has_name(name) {
		return error('module registration name "${name}" is already in use')
	}
	mut seen := []voidptr{}
	if module_contains[T](child, voidptr(module), mut seen) {
		return error('module registration would create a cycle')
	}
	module.children << NamedModule[T]{
		name:   name
		module: child
	}
}

fn (module &Module[T]) collect_named_parameters(prefix string, mut result []NamedParameter[T]) {
	for parameter in module.parameters {
		result << NamedParameter[T]{
			name:  if prefix == '' { parameter.name } else { '${prefix}.${parameter.name}' }
			value: parameter.value
		}
	}
	for child in module.children {
		child_prefix := if prefix == '' { child.name } else { '${prefix}.${child.name}' }
		child.module.collect_named_parameters(child_prefix, mut result)
	}
}

// named_parameters returns local parameters followed by depth-first child
// parameters, joining nested names with dots.
pub fn (module &Module[T]) named_parameters() []NamedParameter[T] {
	mut result := []NamedParameter[T]{}
	module.collect_named_parameters('', mut result)
	return result
}

// parameters returns registered parameter variables without their names.
pub fn (module &Module[T]) parameters() []&autograd.Variable[T] {
	named := module.named_parameters()
	mut result := []&autograd.Variable[T]{cap: named.len}
	for parameter in named {
		result << parameter.value
	}
	return result
}

fn (module &Module[T]) collect_named_buffers(prefix string, mut result []NamedBuffer[T]) {
	for buffer in module.buffers {
		result << NamedBuffer[T]{
			name:  if prefix == '' { buffer.name } else { '${prefix}.${buffer.name}' }
			value: buffer.value
		}
	}
	for child in module.children {
		child_prefix := if prefix == '' { child.name } else { '${prefix}.${child.name}' }
		child.module.collect_named_buffers(child_prefix, mut result)
	}
}

// named_buffers returns local buffers followed by depth-first child buffers,
// joining nested names with dots.
pub fn (module &Module[T]) named_buffers() []NamedBuffer[T] {
	mut result := []NamedBuffer[T]{}
	module.collect_named_buffers('', mut result)
	return result
}

// buffers returns registered tensors without their names.
pub fn (module &Module[T]) buffers() []&vtl.Tensor[T] {
	named := module.named_buffers()
	mut result := []&vtl.Tensor[T]{cap: named.len}
	for buffer in named {
		result << buffer.value
	}
	return result
}
