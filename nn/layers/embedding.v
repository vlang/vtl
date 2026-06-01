module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// EmbeddingLayer maps integer token indices to dense embedding vectors.
//
// Input:    `[batch, seq_len]` — integer indices in `[0, vocab_size)`
// Output:   `[batch, seq_len, embedding_dim]`
//
// Weight shape: `[vocab_size, embedding_dim]`
pub struct EmbeddingLayer[T] {
	vocab_size    int
	embedding_dim int
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
}

// embedding_layer creates an EmbeddingLayer.
pub fn embedding_layer[T](ctx &autograd.Context[T], vocab_size int, embedding_dim int) types.Layer[T] {
	weight := internal.kaiming_uniform[T]([vocab_size, embedding_dim])
	layer := &EmbeddingLayer[T]{
		vocab_size:    vocab_size
		embedding_dim: embedding_dim
		weight:        ctx.variable(weight)
	}
	return types.layer[T](voidptr(layer), embedding_layer_output_shape_dispatch[T],
		embedding_layer_variables_dispatch[T], embedding_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &EmbeddingLayer[T]) output_shape() []int {
	return [layer.embedding_dim]
}

// variables exposes this operation as part of the public API.
pub fn (layer &EmbeddingLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weight]
}

// forward exposes this operation as part of the public API.
pub fn (layer &EmbeddingLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	// input: [batch, seq_len] of integer indices
	// output: [batch, seq_len, embedding_dim]
	output := internal.embedding_forward[T](input.value, layer.weight.value)!
	mut result := input.context.variable(output)
	if input.requires_grad || layer.weight.requires_grad {
		gate := embedding_gate[T](input.value, layer.weight.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn embedding_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&EmbeddingLayer[T](layer)).output_shape() }
}

fn embedding_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&EmbeddingLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn embedding_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&EmbeddingLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// EmbeddingGate defines a public data structure for this module.
pub struct EmbeddingGate[T] {
	input  &vtl.Tensor[T] = unsafe { nil }
	weight &vtl.Tensor[T] = unsafe { nil }
}

// embedding_gate exposes this operation as part of the public API.
pub fn embedding_gate[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) &EmbeddingGate[T] {
	return &EmbeddingGate[T]{
		input:  input
		weight: weight
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &EmbeddingGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.embedding_backward[T](payload.variable.grad, g.input, g.weight)
}

fn embedding_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&EmbeddingGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &EmbeddingGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Embedding', voidptr(g), embedding_gate_backward_dispatch[T],
				result, [a])!
		}
		else {}
	}
}
