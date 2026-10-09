// Compare direct batch reduction with the previous ones-times-gradient path.
// Run from ~/.vmodules with VJOBS=2 and a systemd MemoryMax scope.
// Compile with `v -no-parallel -cc gcc -prod -cflags "-march=native" -o
// /tmp/linear-bias-bench ./vtl/benchmarks/linear_bias_reduction_bench.v`, then
// run `/tmp/linear-bias-bench` separately. Add `-d vsl_blas_generic_cblas` to
// the compile command to compare against the system CBLAS implementation.
module main

import time
import vtl
import vtl.la
import vtl.stats
import vsl.blas as vsl_blas

fn main() {
	println('batch,features,iterations,axis0_sum_ms,ones_matmul_ms,transposed_gemv_ms')
	for shape in [[32, 64], [64, 64], [64, 256], [128, 128], [256, 64], [128, 256], [1024, 256],
		[4096, 512]] {
		batch, features := shape[0], shape[1]
		mut values := []f32{len: batch * features}
		for index in 0 .. values.len {
			values[index] = f32((index * 17) % 101) / 101.0
		}
		gradient := vtl.from_array[f32](values, shape)!
		ones := vtl.ones[f32]([1, batch])
		iterations := 50
		for _ in 0 .. 5 {
			_ = stats.sum_along_axis[f32](gradient, 0, true)!
			_ = la.matmul[f32](ones, gradient)!
			_ = timed_transposed_gemv(gradient, 1)!
		}
		axis_sum := timed_axis0_sum(gradient, iterations)!
		matrix_product := timed_ones_matmul(ones, gradient, iterations)!
		gemv_product := timed_transposed_gemv(gradient, iterations)!
		assert axis_sum.tensor.allclose(matrix_product.tensor, rtol: 1e-5, atol: 1e-5)!
		assert axis_sum.tensor.allclose(gemv_product.tensor, rtol: 1e-5, atol: 1e-5)!
		println('${batch},${features},${iterations},${axis_sum.ms:.4f},${matrix_product.ms:.4f},${gemv_product.ms:.4f}')
	}
}

struct TimedReduction {
	tensor &vtl.Tensor[f32]
	ms     f64
}

fn timed_axis0_sum(gradient &vtl.Tensor[f32], iterations int) !TimedReduction {
	mut result := &vtl.Tensor[f32](unsafe { nil })
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result = stats.sum_along_axis[f32](gradient, 0, true)!
	}
	return TimedReduction{
		tensor: result
		ms:     f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	}
}

fn timed_ones_matmul(ones &vtl.Tensor[f32], gradient &vtl.Tensor[f32], iterations int) !TimedReduction {
	mut result := &vtl.Tensor[f32](unsafe { nil })
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result = la.matmul[f32](ones, gradient)!
	}
	return TimedReduction{
		tensor: result
		ms:     f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	}
}

fn timed_transposed_gemv(gradient &vtl.Tensor[f32], iterations int) !TimedReduction {
	batch := gradient.shape[0]
	features := gradient.size / batch
	ones := []f32{len: batch, init: f32(1)}
	started := time.sys_mono_now()
	mut output := &vtl.Tensor[f32](unsafe { nil })
	for _ in 0 .. iterations {
		output = vtl.zeros[f32]([1, features])
		vsl_blas.sgemv(.trans, batch, features, 1, gradient.data.data, features, ones, 1, 0,
			mut output.data.data, 1)
	}
	return TimedReduction{
		tensor: output
		ms:     f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	}
}
