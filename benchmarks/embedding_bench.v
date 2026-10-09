// Compare contiguous embedding kernels with the coordinate-indexed reference.
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import time
import vtl
import vtl.nn.internal

fn main() {
	batch := 8
	seq_len := 1024
	vocab_size := 2048
	embedding_dim := 64
	iterations := 10
	warmup_iterations := 2

	mut indices := []f64{len: batch * seq_len}
	for i in 0 .. indices.len {
		indices[i] = f64((i * 31) % vocab_size)
	}
	mut weights := []f64{len: vocab_size * embedding_dim}
	for i in 0 .. weights.len {
		weights[i] = f64(i % 101) / 101.0
	}
	mut gradients := []f64{len: batch * seq_len * embedding_dim}
	for i in 0 .. gradients.len {
		gradients[i] = f64(i % 17) / 17.0
	}
	input := vtl.from_array(indices, [batch, seq_len])!
	weight := vtl.from_array(weights, [vocab_size, embedding_dim])!
	grad_out := vtl.from_array(gradients, [batch, seq_len, embedding_dim])!

	for _ in 0 .. warmup_iterations {
		_ = internal.embedding_forward[f64](input, weight)!
		_ = reference_forward(input, weight)!
		_ = internal.embedding_backward[f64](grad_out, input, weight)!
		_ = reference_backward(grad_out, input, weight)!
	}

	fast_forward := timed_forward(input, weight, iterations)!
	reference_forward_time := timed_reference_forward(input, weight, iterations)!
	assert fast_forward.tensor.to_array() == reference_forward(input, weight)!.to_array()
	fast_backward := timed_backward(grad_out, input, weight, iterations)!
	reference_backward_time := timed_reference_backward(grad_out, input, weight, iterations)!
	assert fast_backward.tensors[0].to_array() == reference_backward(grad_out, input, weight)![0].to_array()

	println('operation,tokens,embedding_dim,iterations,method,mean_ms')
	println('forward,${batch * seq_len},${embedding_dim},${iterations},contiguous,${fast_forward.ms:.6f}')
	println('forward,${batch * seq_len},${embedding_dim},${iterations},coordinate_reference,${reference_forward_time:.6f}')
	println('backward,${batch * seq_len},${embedding_dim},${iterations},contiguous,${fast_backward.ms:.6f}')
	println('backward,${batch * seq_len},${embedding_dim},${iterations},coordinate_reference,${reference_backward_time:.6f}')
}

struct TimedForward {
	tensor &vtl.Tensor[f64]
	ms     f64
}

struct TimedBackward {
	tensors []&vtl.Tensor[f64]
	ms      f64
}

fn timed_forward(input &vtl.Tensor[f64], weight &vtl.Tensor[f64], iterations int) !TimedForward {
	mut result := &vtl.Tensor[f64](unsafe { nil })
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result = internal.embedding_forward[f64](input, weight)!
	}
	return TimedForward{
		tensor: result
		ms:     f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	}
}

fn timed_reference_forward(input &vtl.Tensor[f64], weight &vtl.Tensor[f64], iterations int) !f64 {
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		_ = reference_forward(input, weight)!
	}
	return f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
}

fn timed_backward(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64], iterations int) !TimedBackward {
	mut result := []&vtl.Tensor[f64]{}
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result = internal.embedding_backward[f64](grad_out, input, weight)!
	}
	return TimedBackward{
		tensors: result
		ms:      f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	}
}

fn timed_reference_backward(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64], iterations int) !f64 {
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		_ = reference_backward(grad_out, input, weight)!
	}
	return f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
}

fn reference_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch := input.shape[0]
	seq_len := input.shape[1]
	embedding_dim := weight.shape[1]
	mut output := vtl.zeros[T]([batch, seq_len, embedding_dim])
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			idx := int(input.get([b, s]))
			if idx >= 0 && idx < weight.shape[0] {
				for d in 0 .. embedding_dim {
					output.set([b, s, d], weight.get([idx, d]))
				}
			}
		}
	}
	return output
}

fn reference_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	batch := grad_out.shape[0]
	seq_len := grad_out.shape[1]
	embedding_dim := grad_out.shape[2]
	mut d_weight := vtl.zeros_like[T](weight)
	for b in 0 .. batch {
		for s in 0 .. seq_len {
			idx := int(input.get([b, s]))
			if idx >= 0 && idx < weight.shape[0] {
				for d in 0 .. embedding_dim {
					existing := f64(d_weight.get([idx, d]))
					gradient := f64(grad_out.get([b, s, d]))
					d_weight.set([idx, d], vtl.cast[T](existing + gradient))
				}
			}
		}
	}
	return [d_weight]
}
