module internal

import vsl.cuda
import vtl
import vtl.autograd_cuda

// Embedding backward on cuBLAS (ADR-002, phase 3). cuDNN has no scatter-add primitive,
// so the weight gradient is a GEMM with the one-hot index matrix:
//
//   dW[V, D] = onehot[T, V]^T * grad_out[T, D]
//
// where T = batch * seq_len. The one-hot matrix is built on the host, so the path needs
// T * V doubles of extra memory. That cost is accepted for the first version; a scatter
// kernel is the follow-up if profiling shows it matters.

// embedding_cuda_eligible reports whether Embedding backward may run on cuBLAS.
fn embedding_cuda_eligible() bool {
	return autograd_cuda.cuda_backward_enabled()
}

// embedding_backward_cuda_f64 computes the Embedding weight gradient on cuBLAS.
// Out-of-range indices contribute nothing, as on the CPU path.
fn embedding_backward_cuda_f64(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	batch := grad_out.shape[0]
	seq_len := grad_out.shape[1]
	embedding_dim := grad_out.shape[2]
	vocab_size := weight.shape[0]
	tokens := batch * seq_len
	if tokens == 0 || vocab_size == 0 || embedding_dim == 0 {
		return [vtl.zeros_like[f64](weight)]
	}

	indices := input.to_array()
	mut onehot := []f64{len: tokens * vocab_size}
	for token in 0 .. tokens {
		idx := int(indices[token])
		if idx >= 0 && idx < vocab_size {
			onehot[token * vocab_size + idx] = 1.0
		}
	}
	grads := grad_out.to_array()

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}

	d_onehot := device_alloc_bytes(onehot.len * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_onehot)
	}
	d_grad := device_alloc_bytes(grads.len * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_grad)
	}
	d_dweight := device_alloc_bytes(vocab_size * embedding_dim * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dweight)
	}
	device_upload_f64(d_onehot, onehot)!
	device_upload_f64(d_grad, grads)!

	// Column-major view of row-major buffers: grad_out [T, D] is G^T with shape [D, T]
	// (leading dimension D), and the one-hot [T, V] is O^T with shape [V, T] (leading
	// dimension V). The result dW^T with shape [D, V] has the same memory as dW [V, D].
	//   dW^T = G^T * (O^T)^T   with op(B) = transpose.
	alpha := f64(1.0)
	beta := f64(0.0)
	status := C.cublasDgemm_v2(dev.cublas, 0, 1, embedding_dim, vocab_size, tokens, &alpha,
		&f64(d_grad), embedding_dim, &f64(d_onehot), vocab_size, &beta, &f64(d_dweight), embedding_dim)
	if status != cuda.cublas_status_success {
		return error('embedding cuBLAS backward: cublasDgemm: ${cuda.cublas_error(status)}')
	}

	mut dweight_values := []f64{len: vocab_size * embedding_dim}
	device_download_f64(d_dweight, mut dweight_values)!
	return [vtl.from_array(dweight_values, weight.shape)!]
}
