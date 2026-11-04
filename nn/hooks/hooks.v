module hooks

import vtl
import vtl.autograd
import vtl.nn.types

// Module hooks for forward and backward inspection (issue #808).
//
// with_hooks wraps any types.Layer without changing the layer or the Layer type. The wrapper
// is itself a types.Layer, so it can be placed in a Sequential or nested inside another
// wrapper. The registry is not part of the layer's variables or serialized parameters.
//
// Contract:
//   - Forward. Pre-hooks run first, in registration order. Each receives the input produced by
//     the previous hook and may replace it. The wrapped layer then runs. Post-hooks run next,
//     in registration order, each receiving the input the layer saw and the output of the
//     previous hook. A hook error stops the forward pass and is returned unchanged.
//   - Backward. Backward hooks are attached to the module output only when that output
//     requires a gradient. They run in registration order, each receiving the gradient
//     produced by the previous hook, and the last result flows into the layer output. A hook
//     error stops backpropagation and is returned from the backward call.
//   - No gradients. Backward hooks attach only when Variable.is_grad_needed() holds, that is
//     requires_grad is set and the context is not no_grad. Pre- and post-hooks run either way.
//   - Removal. Each registration returns a HookHandle. remove() takes the hook out of the
//     registry, so later passes do not run it. Calling remove() again has no effect.
//   - Order between kinds: forward_pre, the layer, forward, then backward in reverse graph order.

// ForwardPreHook may replace the module input.
pub type ForwardPreHook[T] = fn (input &autograd.Variable[T]) !&autograd.Variable[T]

// ForwardHook receives the input the layer saw and the output, and may replace the output.
pub type ForwardHook[T] = fn (input &autograd.Variable[T], output &autograd.Variable[T]) !&autograd.Variable[T]

// BackwardHook receives the gradient with respect to the module output and may replace it.
pub type BackwardHook[T] = fn (grad &vtl.Tensor[T]) !&vtl.Tensor[T]

// HookKind names the three kinds of registration.
pub enum HookKind {
	forward_pre
	forward
	backward
}

struct PreEntry[T] {
	id   int
	hook ForwardPreHook[T]
}

struct PostEntry[T] {
	id   int
	hook ForwardHook[T]
}

struct BackEntry[T] {
	id   int
	hook BackwardHook[T]
}

// HookRegistry holds the registered hooks of one wrapped module. The wrapper and every
// HookHandle share it, so removal is visible to later passes.
pub struct HookRegistry[T] {
mut:
	next_id int
	pre     []PreEntry[T]
	post    []PostEntry[T]
	back    []BackEntry[T]
}

// HookHandle removes one registration.
pub struct HookHandle[T] {
	registry &HookRegistry[T] = unsafe { nil }
	kind     HookKind
	id       int
}

// remove unregisters the hook. Calling it again has no effect.
pub fn (h HookHandle[T]) remove() {
	if h.registry == unsafe { nil } {
		return
	}
	mut registry := h.registry
	registry.remove(h.kind, h.id)
}

fn (mut r HookRegistry[T]) remove(kind HookKind, id int) {
	match kind {
		.forward_pre {
			r.pre = r.pre.filter(it.id != id)
		}
		.forward {
			r.post = r.post.filter(it.id != id)
		}
		.backward {
			r.back = r.back.filter(it.id != id)
		}
	}
}

// Hooks gives access to the registration methods of a hooked wrapper.
pub struct Hooks[T] {
pub:
	registry &HookRegistry[T]
}

// add_forward_pre_hook registers a pre-hook and returns its handle.
pub fn (h Hooks[T]) add_forward_pre_hook(hook ForwardPreHook[T]) HookHandle[T] {
	mut reg := h.registry
	reg.next_id++
	reg.pre << PreEntry[T]{
		id:   reg.next_id
		hook: hook
	}
	return HookHandle[T]{
		registry: h.registry
		kind:     .forward_pre
		id:       reg.next_id
	}
}

// add_forward_hook registers a post-hook and returns its handle.
pub fn (h Hooks[T]) add_forward_hook(hook ForwardHook[T]) HookHandle[T] {
	mut reg := h.registry
	reg.next_id++
	reg.post << PostEntry[T]{
		id:   reg.next_id
		hook: hook
	}
	return HookHandle[T]{
		registry: h.registry
		kind:     .forward
		id:       reg.next_id
	}
}

// add_backward_hook registers a backward hook and returns its handle.
pub fn (h Hooks[T]) add_backward_hook(hook BackwardHook[T]) HookHandle[T] {
	mut reg := h.registry
	reg.next_id++
	reg.back << BackEntry[T]{
		id:   reg.next_id
		hook: hook
	}
	return HookHandle[T]{
		registry: h.registry
		kind:     .backward
		id:       reg.next_id
	}
}

// HookedModule is the state behind a hooked wrapper: the wrapped layer and its registry.
struct HookedModule[T] {
	inner    types.Layer[T]
	registry &HookRegistry[T]
}

// with_hooks wraps layer. It returns the wrapper, used in place of layer, and the Hooks
// object used to register hooks on it.
pub fn with_hooks[T](layer types.Layer[T]) (types.Layer[T], Hooks[T]) {
	registry := &HookRegistry[T]{}
	module := &HookedModule[T]{
		inner:    layer
		registry: registry
	}
	wrapped := types.layer_with_mode[T](voidptr(module), hooked_output_shape_dispatch[T],
		hooked_variables_dispatch[T], hooked_forward_dispatch[T], hooked_forward_mode_dispatch[T])
	return wrapped, Hooks[T]{
		registry: registry
	}
}

fn hooked_output_shape_dispatch[T](ptr voidptr) []int {
	module := unsafe { &HookedModule[T](ptr) }
	return module.inner.output_shape()
}

fn hooked_variables_dispatch[T](ptr voidptr) []voidptr {
	module := unsafe { &HookedModule[T](ptr) }
	return types.variable_ptrs_to_voidptrs[T](module.inner.variables())
}

fn hooked_forward_dispatch[T](ptr voidptr, input voidptr) !voidptr {
	module := unsafe { &HookedModule[T](ptr) }
	result := module.run(unsafe { &autograd.Variable[T](input) }, true)!
	return voidptr(result)
}

fn hooked_forward_mode_dispatch[T](ptr voidptr, input voidptr, training bool) !voidptr {
	module := unsafe { &HookedModule[T](ptr) }
	result := module.run(unsafe { &autograd.Variable[T](input) }, training)!
	return voidptr(result)
}

// run executes the forward part of the contract, and attaches backward hooks when the
// output requires a gradient.
fn (m &HookedModule[T]) run(input &autograd.Variable[T], training bool) !&autograd.Variable[T] {
	mut x := input
	for entry in m.registry.pre {
		x = entry.hook(x)!
	}
	mut output := m.inner.forward_with_mode(x, training)!
	for entry in m.registry.post {
		output = entry.hook(x, output)!
	}
	if m.registry.back.len > 0 && output.is_grad_needed() {
		mut attached := output.context.variable(output.value, autograd.ContextVariableData{})
		gate := &BackwardHookGate[T]{
			registry: m.registry
		}
		gate.cache(mut attached, output)!
		return attached
	}
	return output
}

// BackwardHookGate runs the registered backward hooks in order.
struct BackwardHookGate[T] {
	registry &HookRegistry[T]
}

// backward applies each registered backward hook to the gradient, in registration order.
pub fn (g &BackwardHookGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	mut grad := payload.variable.grad
	for entry in g.registry.back {
		grad = entry.hook(grad)!
	}
	return [grad]
}

fn backward_hook_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&BackwardHookGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache records the gate in the autograd graph, with the module output as its only parent.
pub fn (g &BackwardHookGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('ModuleBackwardHooks', voidptr(g), backward_hook_gate_backward_dispatch[T],
				result, [a])!
		}
		else {
			return error('BackwardHookGate: argument must be a Variable')
		}
	}
}
