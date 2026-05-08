module layers

import math
import vtl
import vtl.autograd
import vtl.nn.types

pub struct PositionalEncodingLayer[T] {
	max_len    int
	embed_dim  int
pub mut:
	pe         &vtl.Tensor[T] = unsafe { nil }
}

pub fn positional_encoding_layer[T](ctx &autograd.Context[T], embed_dim int, max_len int) types.Layer[T] {
	mut pe_data := []f64{len: max_len * embed_dim}
	for pos in 0 .. max_len {
		for i in 0 .. embed_dim {
			if i % 2 == 0 {
				pe_data[pos * embed_dim + i] = math.sin(f64(pos) / math.pow(10000.0, f64(i) / f64(embed_dim)))
			} else {
				pe_data[pos * embed_dim + i] = math.cos(f64(pos) / math.pow(10000.0, f64(i - 1) / f64(embed_dim)))
			}
		}
	}
	pe := vtl.from_array[T](pe_data.map(vtl.cast[T](it)), [max_len, embed_dim])!
	return types.Layer[T](&PositionalEncodingLayer[T]{
		max_len: max_len, embed_dim: embed_dim, pe: pe
	})
}

pub fn (layer &PositionalEncodingLayer[T]) output_shape() []int { return []int{layer.embed_dim} }
pub fn (layer &PositionalEncodingLayer[T]) variables() []&autograd.Variable[T] { return [] }

pub fn (layer &PositionalEncodingLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	seq_len := input.value.shape[1]
	batch_size := input.value.shape[0]
	_ := batch_size
	pe_slice := layer.pe.slice(0, seq_len)!
	output := input.value.map([pe_slice], fn [T](vals []T, i []int) T {
		return vals[0] + vals[1]
	})!
	return input.context.variable(output)
}
