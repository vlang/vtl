module optimizers

import math
import os
import vtl
import vtl.autograd

fn test_adam_f32_vulkan_matches_cpu() ! {
	if os.getenv('VTL_USE_VULKAN') != '1' {
		return
	}
	c := autograd.ctx[f32]()
	mut p_cpu := c.variable(vtl.from_1d([f32(5.0), f32(5.0)])!)
	mut p_gpu := c.variable(vtl.from_1d([f32(5.0), f32(5.0)])!)
	p_cpu.grad = vtl.from_1d([f32(1.0), f32(2.0)])!
	p_gpu.grad = vtl.from_1d([f32(1.0), f32(2.0)])!
	m_cpu := vtl.zeros_like[f32](p_cpu.grad)
	v_cpu := vtl.zeros_like[f32](p_cpu.grad)
	mut m_gpu := vtl.zeros_like[f32](p_gpu.grad)
	mut v_gpu := vtl.zeros_like[f32](p_gpu.grad)
	step := AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.001
		epsilon: 1e-8
	}
	grad := p_cpu.grad.to_array()
	mut th_cpu := p_cpu.value.to_array()
	mut ma := m_cpu.to_array()
	mut va := v_cpu.to_array()
	adam_step_f32_cpu(grad, mut th_cpu, mut ma, mut va, step)
	assert try_adam_update_f32_vulkan(mut p_gpu, mut m_gpu, mut v_gpu, step)
	for i in 0 .. th_cpu.len {
		assert math.abs(th_cpu[i] - p_gpu.value.get_nth(i)) < 1e-3
	}
}
