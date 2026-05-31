module autograd

import vsl.cuda
import vtl
import vtl.storage

// DeviceGpuChain holds reusable device buffers for chained Linear forwards (Phase 2).
@[heap]
struct DeviceGpuChain {
mut:
	d_a GpuDevBuf
	d_b GpuDevBuf
	d_c GpuDevBuf
	// Pending CudaTensor view over d_c (ownership stays in d_c until taken).
	pending &vtl.CudaTensor[f64] = unsafe { nil }
}

struct GpuDevBuf {
mut:
	ptr  voidptr
	size int
}

fn (mut b GpuDevBuf) ensure(count int) ! {
	sz := int(sizeof(f64)) * count
	if b.ptr != unsafe { nil } && b.size >= sz {
		return
	}
	if b.ptr != unsafe { nil } {
		C.cudaFree(b.ptr)
	}
	mut ptr := unsafe { nil }
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('GpuDevBuf.ensure: cudaMalloc status ${status}')
	}
	b.ptr = ptr
	b.size = sz
}

fn (b &GpuDevBuf) release() {
	if b.ptr != unsafe { nil } {
		C.cudaFree(b.ptr)
	}
}

fn (mut chain DeviceGpuChain) release_all() {
	chain.d_a.release()
	chain.d_b.release()
	chain.d_c.release()
	if chain.pending != unsafe { nil } {
		chain.pending.release()
		chain.pending = unsafe { nil }
	}
}

fn (mut s DeviceSession) gpu_chain_mut() &DeviceGpuChain {
	if s.gpu_chain == unsafe { nil } {
		s.gpu_chain = voidptr(&DeviceGpuChain{})
	}
	return unsafe { &DeviceGpuChain(s.gpu_chain) }
}

pub fn (mut s DeviceSession) reset_gpu_chain() {
	if s.gpu_chain != unsafe { nil } {
		mut chain := unsafe { &DeviceGpuChain(s.gpu_chain) }
		chain.release_all()
		s.gpu_chain = unsafe { nil }
	}
}

// take_chain_activation_impl moves the pending GPU tensor to the caller (Variable).
fn take_chain_activation_impl(mut s DeviceSession) !&vtl.CudaTensor[f64] {
	if s.gpu_chain == unsafe { nil } {
		return error('no gpu chain')
	}
	mut chain := s.gpu_chain_mut()
	if chain.pending == unsafe { nil } {
		return error('no pending gpu activation')
	}
	t := chain.pending
	chain.pending = unsafe { nil }
	return t
}

// linear_forward_gpu_chain runs GEMM on device; reuses input GPU activation when provided.
fn (mut s DeviceSession) linear_forward_gpu_chain(x &vtl.Tensor[f64], weights &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], input_gpu voidptr) !&vtl.Tensor[f64] {
	if !gpu_activations_enabled() {
		return error('gpu chain disabled')
	}
	dev := cuda.get_default_device()!

	m := x.shape[0]
	k := x.shape[1]
	n := weights.shape[0]

	mut chain := s.gpu_chain_mut()

	// Free previous pending view before overwriting d_c.
	if chain.pending != unsafe { nil } {
		chain.pending.release()
		chain.pending = unsafe { nil }
	}

	w_arr := weights.to_array()
	wt_row := transpose_weights_row(w_arr, n, k)

	// d_a: reuse input GPU activation or upload x.
	chain.d_a.ensure(m * k)!
	if input_gpu != unsafe { nil } {
		in_ct := unsafe { &vtl.CudaTensor[f64](input_gpu) }
		if in_ct.shape != [m, k] {
			return error('gpu chain: input activation shape mismatch')
		}
		// cudaMemcpy kind 4 = device-to-device (see vsl/cuda/_cfun.c.v).
		status := C.cudaMemcpy(chain.d_a.ptr, in_ct.data.ptr, int(sizeof(f64)) * m * k, 4)
		if status != 0 {
			return error('gpu chain: D2D input copy status ${status}')
		}
	} else {
		x_arr := x.to_array()
		x_col := stage_col_major(mut s.gemm_x_col, x_arr, m, k)
		status := C.cudaMemcpy(chain.d_a.ptr, x_col.data, int(sizeof(f64)) * m * k,
			cuda.cuda_memcpy_host_to_device)
		if status != 0 {
			return error('gpu chain: H2D x status ${status}')
		}
	}

	w_col := stage_col_major(mut s.gemm_w_col, wt_row, k, n)
	chain.d_b.ensure(k * n)!
	status_b := C.cudaMemcpy(chain.d_b.ptr, w_col.data, int(sizeof(f64)) * k * n,
		cuda.cuda_memcpy_host_to_device)
	if status_b != 0 {
		return error('gpu chain: H2D w status ${status_b}')
	}

	chain.d_c.ensure(m * n)!
	alpha := f64(1.0)
	beta := f64(0.0)
	status := C.cublasDgemm_v2(dev.cublas, 0, 0, m, n, k, &alpha, &f64(chain.d_a.ptr), m,
		&f64(chain.d_b.ptr), k, &beta, &f64(chain.d_c.ptr), m)
	if status != cuda.cublas_status_success {
		return error('gpu chain: cublasDgemm failed: ${cuda.cublas_error(status)}')
	}

	// One D2H per layer for CPU autograd + bias; input skips H2D when chained.
	s.gemm_out_row = resize_f64(mut s.gemm_out_row, m * n)
	status_dl := C.cudaMemcpy(s.gemm_out_row.data, chain.d_c.ptr, int(sizeof(f64)) * m * n,
		cuda.cuda_memcpy_device_to_host)
	if status_dl != 0 {
		return error('gpu chain: D2H out status ${status_dl}')
	}

	b_arr := bias.to_array()
	for i in 0 .. s.gemm_out_row.len {
		col_idx := i % n
		s.gemm_out_row[i] += b_arr[col_idx]
	}

	// Hand off d_c buffer to pending activation (chain.d_c no longer owns the ptr).
	out_ptr := chain.d_c.ptr
	chain.d_c.ptr = unsafe { nil }
	chain.d_c.size = 0
	chain.pending = &vtl.CudaTensor[f64]{
		data: &storage.CudaStorage[f64]{
			device: dev
			ptr:    out_ptr
			size:   int(sizeof(f64)) * m * n
			count:  m * n
		}
		memory:  .row_major
		size:    m * n
		shape:   [m, n]
		strides: [n, 1]
	}

	return vtl.from_array(s.gemm_out_row, [m, n])!
}
