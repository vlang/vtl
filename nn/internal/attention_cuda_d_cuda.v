module internal

import math
import vsl.cuda
import vtl
import vtl.autograd_cuda

// Multi-head attention backward on cuBLAS (ADR-002, phase 3). Every matrix product is a
// cuBLAS GEMM, looped over (batch, head) for the per-head products. The Q/K/V projections
// and the output projection are single GEMMs over the flattened [batch * seq_len, embed]
// view.
//
// The softmax backward is elementwise per row. It runs on the host in this version, which
// downloads and uploads the [batch, heads, seq, seq] score gradients once. cudnnSoftmaxBackward
// is the follow-up. The per-(batch, head) GEMM loop is also a follow-up: strided-batched
// GEMM would replace it.

// attention_cuda_eligible reports whether attention backward may run on cuBLAS.
pub fn attention_cuda_eligible() bool {
	return autograd_cuda.cuda_backward_enabled()
}

// attn_gemm computes the row-major product C = op(A) * op(B) with cuBLAS. A, B and C are
// device pointers to row-major blocks; lda, ldb and ldc are the row-major leading
// dimensions (the width of the full matrix that holds each block). op(A) is [m x k] and
// op(B) is [k x n]; C is [m x n].
//
// cuBLAS works in column-major order, and a row-major matrix is its transpose in
// column-major order. So C = op(A) op(B) is computed as C^T = op(B)^T op(A)^T, with the
// operands swapped and the transpose flags kept.
fn attn_gemm(handle voidptr, trans_a bool, trans_b bool, m int, n int, k int, alpha f64, a voidptr, lda int, b voidptr, ldb int, beta f64, c voidptr, ldc int) ! {
	op_a := if trans_a { 1 } else { 0 }
	op_b := if trans_b { 1 } else { 0 }
	status := C.cublasDgemm_v2(handle, op_b, op_a, n, m, k, &alpha, &f64(b), ldb, &f64(a), lda,
		&beta, &f64(c), ldc)
	if status != cuda.cublas_status_success {
		return error('attention cuBLAS backward: cublasDgemm: ${cuda.cublas_error(status)}')
	}
}

// attn_offset returns base advanced by count float64 elements.
fn attn_offset(base voidptr, count int) voidptr {
	return unsafe { voidptr(u64(base) + u64(count) * u64(sizeof(f64))) }
}

// attn_upload uploads a host slice into a new device buffer and returns the buffer.
fn attn_upload(values []f64) !voidptr {
	ptr := device_alloc_bytes(values.len * int(sizeof(f64)))!
	device_upload_f64(ptr, values) or {
		C.cudaFree(ptr)
		return err
	}
	return ptr
}

// attention_backward_cuda_f64 returns [d_input, d_w_q, d_w_k, d_w_v, d_w_o], matching the
// CPU path in AttentionGate.backward. q, k, v, merged and attn_weights are the forward
// intermediates cached on the gate.
pub fn attention_backward_cuda_f64(grad &vtl.Tensor[f64], input &vtl.Tensor[f64], w_q &vtl.Tensor[f64], w_k &vtl.Tensor[f64], w_v &vtl.Tensor[f64], w_o &vtl.Tensor[f64], q &vtl.Tensor[f64], k &vtl.Tensor[f64], v &vtl.Tensor[f64], attn_weights &vtl.Tensor[f64], merged &vtl.Tensor[f64], num_heads int, head_dim int) ![]&vtl.Tensor[f64] {
	batch := input.shape[0]
	seq := input.shape[1]
	embed := input.shape[2]
	rows := batch * seq
	heads := num_heads
	scale := 1.0 / math.sqrt(f64(head_dim))

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}
	handle := dev.cublas

	d_x := attn_upload(input.to_array())!
	defer {
		C.cudaFree(d_x)
	}
	d_grad := attn_upload(grad.to_array())!
	defer {
		C.cudaFree(d_grad)
	}
	d_wq := attn_upload(w_q.to_array())!
	defer {
		C.cudaFree(d_wq)
	}
	d_wk := attn_upload(w_k.to_array())!
	defer {
		C.cudaFree(d_wk)
	}
	d_wv := attn_upload(w_v.to_array())!
	defer {
		C.cudaFree(d_wv)
	}
	d_wo := attn_upload(w_o.to_array())!
	defer {
		C.cudaFree(d_wo)
	}
	d_q := attn_upload(q.to_array())!
	defer {
		C.cudaFree(d_q)
	}
	d_k := attn_upload(k.to_array())!
	defer {
		C.cudaFree(d_k)
	}
	d_v := attn_upload(v.to_array())!
	defer {
		C.cudaFree(d_v)
	}
	d_merged := attn_upload(merged.to_array())!
	defer {
		C.cudaFree(d_merged)
	}
	d_attn := attn_upload(attn_weights.to_array())!
	defer {
		C.cudaFree(d_attn)
	}

	// Device buffers that receive results or intermediates.
	bytes_rows := rows * embed * int(sizeof(f64))
	bytes_scores := batch * heads * seq * seq * int(sizeof(f64))
	bytes_weights := embed * embed * int(sizeof(f64))
	d_dmerged := device_alloc_bytes(bytes_rows)!
	defer {
		C.cudaFree(d_dmerged)
	}
	d_da := device_alloc_bytes(bytes_scores)!
	defer {
		C.cudaFree(d_da)
	}
	d_ds := device_alloc_bytes(bytes_scores)!
	defer {
		C.cudaFree(d_ds)
	}
	d_dq := device_alloc_bytes(bytes_rows)!
	defer {
		C.cudaFree(d_dq)
	}
	d_dk := device_alloc_bytes(bytes_rows)!
	defer {
		C.cudaFree(d_dk)
	}
	d_dv := device_alloc_bytes(bytes_rows)!
	defer {
		C.cudaFree(d_dv)
	}
	d_dinput := device_alloc_bytes(bytes_rows)!
	defer {
		C.cudaFree(d_dinput)
	}
	d_dwq := device_alloc_bytes(bytes_weights)!
	defer {
		C.cudaFree(d_dwq)
	}
	d_dwk := device_alloc_bytes(bytes_weights)!
	defer {
		C.cudaFree(d_dwk)
	}
	d_dwv := device_alloc_bytes(bytes_weights)!
	defer {
		C.cudaFree(d_dwv)
	}
	d_dwo := device_alloc_bytes(bytes_weights)!
	defer {
		C.cudaFree(d_dwo)
	}

	// Output projection: d_merged = grad * W_o^T, then dW_o = merged^T * grad.
	attn_gemm(handle, false, true, rows, embed, embed, 1.0, d_grad, embed, d_wo, embed, 0.0,
		d_dmerged, embed)!
	attn_gemm(handle, true, false, embed, embed, rows, 1.0, d_merged, embed, d_grad, embed, 0.0,
		d_dwo, embed)!

	// Per (batch, head): dA = dC * V^T, dV = A^T * dC. A is the cached softmax output.
	for b in 0 .. batch {
		for h in 0 .. heads {
			block := b * seq * embed + h * head_dim
			scores := (b * heads + h) * seq * seq
			dc := attn_offset(d_dmerged, block)
			vb := attn_offset(d_v, block)
			dv := attn_offset(d_dv, block)
			a := attn_offset(d_attn, scores)
			da := attn_offset(d_da, scores)
			attn_gemm(handle, false, true, seq, seq, head_dim, 1.0, dc, embed, vb, embed, 0.0, da,
				seq)!
			attn_gemm(handle, true, false, seq, head_dim, seq, 1.0, a, seq, dc, embed, 0.0, dv,
				embed)!
		}
	}

	// Softmax backward per row on the host: dS = A * (dA - sum_j dA * A).
	mut da_values := []f64{len: batch * heads * seq * seq}
	device_download_f64(d_da, mut da_values)!
	attn_values := attn_weights.to_array()
	mut ds_values := []f64{len: da_values.len}
	for row in 0 .. da_values.len / seq {
		start := row * seq
		mut dot := f64(0)
		for j in 0 .. seq {
			dot += da_values[start + j] * attn_values[start + j]
		}
		for j in 0 .. seq {
			ds_values[start + j] = attn_values[start + j] * (da_values[start + j] - dot)
		}
	}
	device_upload_f64(d_ds, ds_values)!

	// Per (batch, head): dQ = scale * dS * K and dK = scale * dS^T * Q.
	for b in 0 .. batch {
		for h in 0 .. heads {
			block := b * seq * embed + h * head_dim
			scores := (b * heads + h) * seq * seq
			ds := attn_offset(d_ds, scores)
			kb := attn_offset(d_k, block)
			qb := attn_offset(d_q, block)
			dq := attn_offset(d_dq, block)
			dk := attn_offset(d_dk, block)
			attn_gemm(handle, false, false, seq, head_dim, seq, scale, ds, seq, kb, embed, 0.0, dq,
				embed)!
			attn_gemm(handle, true, false, seq, head_dim, seq, scale, ds, seq, qb, embed, 0.0, dk,
				embed)!
		}
	}

	// Projection gradients: dW = X^T * dP for P in {Q, K, V}, and
	// d_input = dQ * W_q^T + dK * W_k^T + dV * W_v^T.
	attn_gemm(handle, true, false, embed, embed, rows, 1.0, d_x, embed, d_dq, embed, 0.0, d_dwq,
		embed)!
	attn_gemm(handle, true, false, embed, embed, rows, 1.0, d_x, embed, d_dk, embed, 0.0, d_dwk,
		embed)!
	attn_gemm(handle, true, false, embed, embed, rows, 1.0, d_x, embed, d_dv, embed, 0.0, d_dwv,
		embed)!
	attn_gemm(handle, false, true, rows, embed, embed, 1.0, d_dq, embed, d_wq, embed, 0.0,
		d_dinput, embed)!
	attn_gemm(handle, false, true, rows, embed, embed, 1.0, d_dk, embed, d_wk, embed, 1.0,
		d_dinput, embed)!
	attn_gemm(handle, false, true, rows, embed, embed, 1.0, d_dv, embed, d_wv, embed, 1.0,
		d_dinput, embed)!

	mut dinput_values := []f64{len: rows * embed}
	device_download_f64(d_dinput, mut dinput_values)!
	mut dwq_values := []f64{len: embed * embed}
	device_download_f64(d_dwq, mut dwq_values)!
	mut dwk_values := []f64{len: embed * embed}
	device_download_f64(d_dwk, mut dwk_values)!
	mut dwv_values := []f64{len: embed * embed}
	device_download_f64(d_dwv, mut dwv_values)!
	mut dwo_values := []f64{len: embed * embed}
	device_download_f64(d_dwo, mut dwo_values)!

	return [
		vtl.from_array(dinput_values, input.shape)!,
		vtl.from_array(dwq_values, w_q.shape)!,
		vtl.from_array(dwk_values, w_k.shape)!,
		vtl.from_array(dwv_values, w_v.shape)!,
		vtl.from_array(dwo_values, w_o.shape)!,
	]
}
