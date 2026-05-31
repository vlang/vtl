module autograd

import os
import vtl
import math

fn test_ctx_f64_has_device_session() {
	c := ctx[f64]()
	assert c.device_session != unsafe { nil }
}

fn test_device_session_cpu_fallback_without_cuda() ! {
	mut s := new_device_session()
	s.init_device()
	x := vtl.from_array([1.0, 2.0], [1, 2])!
	w := vtl.from_array([0.5, 0.5, 0.1, 0.2], [2, 2])!
	b := vtl.from_array([0.0, 0.0], [1, 2])!
	// Without VTL_USE_CUDA / -d cuda, linear_forward_f64 errors → use CPU helper
	cpu := linear_forward_f64_cpu(x, w, b)!
	assert cpu.shape == [1, 2]
}

fn test_device_session_reuses_buffers_on_second_forward() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	mut s := new_device_session()
	s.init_device()
	if !s.enabled {
		return
	}
	x := vtl.from_array([1.0, 2.0, 3.0], [1, 3])!
	w := vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3])!
	b := vtl.from_array([0.01, 0.02], [1, 2])!

	out1 := s.linear_forward_f64(x, w, b, unsafe { nil })!
	len_after_first := s.gemm_out_row.len
	out2 := s.linear_forward_f64(x, w, b, unsafe { nil })!
	assert s.gemm_out_row.len == len_after_first, 'output buffer should be reused, not grow unbounded'
	cpu := linear_forward_f64_cpu(x, w, b)!
	for i in 0 .. out2.size {
		diff := math.abs(f64(out2.get_nth(i)) - f64(cpu.get_nth(i)))
		assert diff < 1e-9, 'session forward mismatch at ${i}'
	}
	_ = out1
}

fn test_gpu_activation_chain_skips_without_env() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	mut s := new_device_session()
	s.init_device()
	if !s.enabled {
		return
	}
	assert !gpu_activations_enabled(), 'without VTL_GPU_ACTIVATIONS chain is off'
	x := vtl.from_array([1.0, 2.0], [1, 2])!
	w := vtl.from_array([0.5, 0.5, 0.1, 0.2], [2, 2])!
	b := vtl.from_array([0.0, 0.0], [1, 2])!
	out := s.linear_forward_f64(x, w, b, unsafe { nil })!
	assert out.shape == [1, 2]
}
