module layers

import math
import vtl
import vtl.autograd
import vtl.nn.types

// PositionalEncodingLayer adds fixed sinusoidal positional encodings to an embedding.
//
// Does not contain learnable parameters. Encodings follow the original Transformer
// formulation (Attention is All You Need, §3.5).
//
// Input:    `[batch, seq_len, embed_dim]`
// Output:   `[batch, seq_len, embed_dim]` — input + positional encoding
pub struct PositionalEncodingLayer[T] {
	max_len   int
	embed_dim int
pub mut:
	pe &vtl.Tensor[T] = unsafe { nil }
}

// positional_encoding_layer creates a PositionalEncodingLayer.
pub fn positional_encoding_layer[T](ctx &autograd.Context[T], embed_dim int, max_len int) !types.Layer[T] {
	mut pe_data := []f64{len: max_len * embed_dim}
	for pos in 0 .. max_len {
		for i in 0 .. embed_dim {
			if i % 2 == 0 {
				pe_data[pos * embed_dim + i] = math.sin(f64(pos) / math.pow(10000.0,
					f64(i) / f64(embed_dim)))
			} else {
				pe_data[pos * embed_dim + i] = math.cos(f64(pos) / math.pow(10000.0,
					f64(i - 1) / f64(embed_dim)))
			}
		}
	}
	pe := vtl.from_array[T](pe_data.map(vtl.cast[T](it)), [max_len, embed_dim])!
	return types.Layer[T](&PositionalEncodingLayer[T]{
		max_len:   max_len
		embed_dim: embed_dim
		pe:        pe
	})
}

pub fn (layer &PositionalEncodingLayer[T]) output_shape() []int {
	return [layer.embed_dim]
}

pub fn (layer &PositionalEncodingLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &PositionalEncodingLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	// input: [batch, seq_len, embed_dim]
	batch := input.value.shape[0]
	seq_len := input.value.shape[1]
	// Add positional encoding element-wise (broadcast over batch)
	mut out_data := []f64{len: batch * seq_len * layer.embed_dim}
	for b in 0 .. batch {
		for p in 0 .. seq_len {
			for e in 0 .. layer.embed_dim {
				pe_val := if p < layer.max_len { f64(layer.pe.get([p, e])) } else { f64(0) }
				inp_val := f64(input.value.get([b, p, e]))
				out_data[b * seq_len * layer.embed_dim + p * layer.embed_dim + e] = inp_val + pe_val
			}
		}
	}
	output := vtl.from_array(out_data.map(vtl.cast[T](it)), [batch, seq_len, layer.embed_dim])!
	return input.context.variable(output)
}
