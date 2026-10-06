// End-to-end resident-buffer Vulkan f32 GEMM benchmark.
// Run from ~/.vmodules with `v -prod -d vulkan run ...`.
module main

import time
import vtl
import vtl.storage
import vsl.vulkan

fn main() {
	mut dev := vulkan.new_device() or {
		eprintln('Vulkan device unavailable: ${err}')
		return
	}
	defer {
		dev.release() or {}
	}
	println('VTL Vulkan GEMM f32 on ${dev.gpu_name()}')
	println('size,iterations,mean_ms,checksum')
	for n in [128, 256, 512, 1024, 2048] {
		bench_matmul(dev, n) or {
			eprintln('Vulkan GEMM ${n}x${n} failed: ${err}')
			return
		}
	}
}

fn bench_matmul(dev &vulkan.Device, n int) ! {
	mut a_values := []f32{len: n * n}
	mut b_values := []f32{len: n * n}
	for i in 0 .. n {
		for j in 0 .. n {
			a_values[i * n + j] = f32((i * 17 + j * 13) % 997 + 1) / 997.0
			b_values[i * n + j] = f32((i * 7 + j * 19) % 991 + 1) / 991.0
		}
	}
	a := vtl.from_array(a_values, [n, n])!
	b := vtl.from_array(b_values, [n, n])!
	params := storage.vulkan_params_for_device(dev)
	a_gpu := a.vulkan(params)!
	defer {
		a_gpu.release()
	}
	b_gpu := b.vulkan(params)!
	defer {
		b_gpu.release()
	}
	mut c_gpu := vtl.vulkan_tensor_zeros_f32([n, n], dev)!
	defer {
		c_gpu.release()
	}
	for _ in 0 .. 3 {
		vtl.gemm_vulkan(c_gpu, a_gpu, b_gpu)!
	}
	iterations := 10
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		vtl.gemm_vulkan(c_gpu, a_gpu, b_gpu)!
	}
	mean_ms := f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
	cpu_result := c_gpu.cpu()!
	checksum := cpu_result.get_nth[f32](n * n / 2 + n / 2)
	println('${n},${iterations},${mean_ms:.3f},${checksum:.6f}')
}
