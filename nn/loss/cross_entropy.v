module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.internal

// CrossEntropyLoss combines LogSoftmax + NLLLoss for unbatched, batched, and
// spatial logits. Reduction, class weights, and ignored indices match PyTorch.
pub struct CrossEntropyLoss[T] {
pub:
	weight          &vtl.Tensor[T] = unsafe { nil }
	ignore_index    int            = -100
	reduction       string         = 'mean' // 'mean' | 'sum' | 'none'
	label_smoothing f64
}

// CrossEntropyLossOptions configures ignored class indices, reduction, and smoothing.
@[params]
pub struct CrossEntropyLossOptions {
pub:
	ignore_index    int    = -100
	reduction       string = 'mean'
	label_smoothing f64
}

// cross_entropy_loss exposes this operation as part of the public API.
pub fn cross_entropy_loss[T]() types.Loss[T] {
	return cross_entropy_loss_with_options[T](unsafe { nil }, CrossEntropyLossOptions{})
}

// cross_entropy_loss_with_options creates a weighted cross-entropy objective.
pub fn cross_entropy_loss_with_options[T](weight &vtl.Tensor[T], options CrossEntropyLossOptions) types.Loss[T] {
	concrete := &CrossEntropyLoss[T]{
		weight:          weight
		ignore_index:    options.ignore_index
		reduction:       options.reduction
		label_smoothing: options.label_smoothing
	}
	return types.loss[T](voidptr(concrete), cross_entropy_loss_loss_dispatch[T])
}

// CrossEntropyIndexLoss adapts integer class-index targets to CrossEntropyLoss.
pub struct CrossEntropyIndexLoss[T] {
	inner &CrossEntropyLoss[T] = unsafe { nil }
}

// cross_entropy_loss_indices creates an objective that accepts Tensor[int] targets.
pub fn cross_entropy_loss_indices[T](weight &vtl.Tensor[T], options CrossEntropyLossOptions) CrossEntropyIndexLoss[T] {
	return CrossEntropyIndexLoss[T]{
		inner: &CrossEntropyLoss[T]{
			weight:          weight
			ignore_index:    options.ignore_index
			reduction:       options.reduction
			label_smoothing: options.label_smoothing
		}
	}
}

// loss exposes this operation as part of the public API.
pub fn (l &CrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.cross_entropy_with_options[T](input.value, target, l.weight, l.ignore_index,
		l.reduction, l.label_smoothing)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := cross_entropy_loss_gate_with_options[T](input.value, target, l.weight, l.ignore_index,
			l.reduction, l.label_smoothing)
		gate.cache(mut result, input)!
	}
	return result
}

// loss computes CrossEntropyLoss from integer class-index targets.
pub fn (l &CrossEntropyIndexLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[int]) !&autograd.Variable[T] {
	mut target_values := []T{len: target.size()}
	for i in 0 .. target.size() {
		target_values[i] = vtl.cast[T](target.get_nth(i))
	}
	typed_target := vtl.from_array(target_values, target.shape.clone())!
	return l.inner.loss(input, typed_target)
}

fn cross_entropy_loss_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&CrossEntropyLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}

// CrossEntropyLossGate defines a public data structure for this module.
pub struct CrossEntropyLossGate[T] {
pub:
	input           &vtl.Tensor[T] = unsafe { nil }
	target          &vtl.Tensor[T] = unsafe { nil }
	weight          &vtl.Tensor[T] = unsafe { nil }
	ignore_index    int            = -100
	reduction       string         = 'mean'
	label_smoothing f64
}

// cross_entropy_loss_gate exposes this operation as part of the public API.
pub fn cross_entropy_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &CrossEntropyLossGate[T] {
	return cross_entropy_loss_gate_with_options[T](input, target, unsafe { nil }, -100, 'mean', 0)
}

// cross_entropy_loss_gate_with_options creates the configured backward gate.
pub fn cross_entropy_loss_gate_with_options[T](input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string, label_smoothing f64) &CrossEntropyLossGate[T] {
	return &CrossEntropyLossGate[T]{
		input:           input
		target:          target
		weight:          weight
		ignore_index:    ignore_index
		reduction:       reduction
		label_smoothing: label_smoothing
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &CrossEntropyLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.cross_entropy_backward_with_options[T](gradient, g.input, g.target, g.weight,
		g.ignore_index, g.reduction, g.label_smoothing)!
	return [r0]
}

fn cross_entropy_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&CrossEntropyLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &CrossEntropyLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('CrossEntropy', voidptr(g),
				cross_entropy_loss_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('CrossEntropy: cache: invalid argument')
		}
	}
}
