module autograd

// GPU-accelerated backward gates for elementwise activation functions.
// Compile with -d vulkan. Each gate stores the VulkanStorageParams (device)
// used during the forward pass so that backward uses the same device.
// Returns CPU Tensor[T] to conform to the Gate[T] interface.

import vtl
import storage
import vsl.vulkan
import math

// ReLUGateVulkan: d_relu = grad * (x > 0)
pub struct ReLUGateVulkan[T] {
pub:
	a      &Variable[T]              = unsafe { nil }
	params storage.VulkanStorageParams
}

pub fn relu_gate_vulkan[T](a &Variable[T], params storage.VulkanStorageParams) &ReLUGateVulkan[T] {
	return &ReLUGateVulkan[T]{a: a, params: params}
}

pub fn (g &ReLUGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	input := g.a.value
	n := input.size
	$if T is f32 {
		dev := g.params.device
		size := vulkan.DeviceSize(u64(n) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		vk_input := input.vulkan(g.params)!
		defer { vk_input.release() or {} }
		mut out_buf := dev.buffer(size)!
		defer { out_buf.release() }
		vulkan.d_relu(dev, out_buf, vk_grad.data.data, vk_input.data.data)!
		mut raw := []u8{len: n * 4}
		raw = out_buf.store(mut raw)!
		mut vals := []T{len: n}
		for i in 0 .. n {
			unsafe { vals[i] = T(*(&f32(&raw[i * 4]))) }
		}
		r0 := vtl.from_array[T](vals, input.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		zero := vtl.cast[T](0)
		r0 := gradient.nmap([input], fn [zero] [T](vals []T, _ []int) T {
			return if vals[1] > zero { vals[0] } else { zero }
		})!
		return [r0]
	}
}

pub fn (g &ReLUGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('ReLUVulkan', g, result, [a])!
		}
		else {
			return error('ReLUGateVulkan: a must be a Variable')
		}
	}
}

// SigmoidGateVulkan: d_sigmoid = grad * s * (1 - s), s = sigmoid(x)
pub struct SigmoidGateVulkan[T] {
pub:
	sigmoid_out &vtl.Tensor[T]       = unsafe { nil }
	params      storage.VulkanStorageParams
}

pub fn sigmoid_gate_vulkan[T](sigmoid_out &vtl.Tensor[T], params storage.VulkanStorageParams) &SigmoidGateVulkan[T] {
	return &SigmoidGateVulkan[T]{sigmoid_out: sigmoid_out, params: params}
}

pub fn (g &SigmoidGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	s := g.sigmoid_out
	n := s.size
	$if T is f32 {
		dev := g.params.device
		size := vulkan.DeviceSize(u64(n) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		vk_s := s.vulkan(g.params)!
		defer { vk_s.release() or {} }
		mut out_buf := dev.buffer(size)!
		defer { out_buf.release() }
		vulkan.d_sigmoid(dev, out_buf, vk_grad.data.data, vk_s.data.data)!
		mut raw := []u8{len: n * 4}
		raw = out_buf.store(mut raw)!
		mut vals := []T{len: n}
		for i in 0 .. n {
			unsafe { vals[i] = T(*(&f32(&raw[i * 4]))) }
		}
		r0 := vtl.from_array[T](vals, s.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		one := vtl.cast[T](1)
		r0 := gradient.nmap([s], fn [one] [T](vals []T, _ []int) T {
			return vals[0] * vals[1] * (one - vals[1])
		})!
		return [r0]
	}
}

pub fn (g &SigmoidGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('SigmoidVulkan', g, result, [a])!
		}
		else {
			return error('SigmoidGateVulkan: a must be a Variable')
		}
	}
}

// TanhGateVulkan: d_tanh = grad * (1 - tanh(x)^2), tanh_out = tanh(x)
pub struct TanhGateVulkan[T] {
pub:
	tanh_out &vtl.Tensor[T]          = unsafe { nil }
	params   storage.VulkanStorageParams
}

pub fn tanh_gate_vulkan[T](tanh_out &vtl.Tensor[T], params storage.VulkanStorageParams) &TanhGateVulkan[T] {
	return &TanhGateVulkan[T]{tanh_out: tanh_out, params: params}
}

pub fn (g &TanhGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	t_out := g.tanh_out
	n := t_out.size
	$if T is f32 {
		dev := g.params.device
		size := vulkan.DeviceSize(u64(n) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		vk_t := t_out.vulkan(g.params)!
		defer { vk_t.release() or {} }
		mut out_buf := dev.buffer(size)!
		defer { out_buf.release() }
		vulkan.d_tanh(dev, out_buf, vk_grad.data.data, vk_t.data.data)!
		mut raw := []u8{len: n * 4}
		raw = out_buf.store(mut raw)!
		mut vals := []T{len: n}
		for i in 0 .. n {
			unsafe { vals[i] = T(*(&f32(&raw[i * 4]))) }
		}
		r0 := vtl.from_array[T](vals, t_out.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		one := vtl.cast[T](1)
		r0 := gradient.nmap([t_out], fn [one] [T](vals []T, _ []int) T {
			return vals[0] * (one - vals[1] * vals[1])
		})!
		return [r0]
	}
}

pub fn (g &TanhGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('TanhVulkan', g, result, [a])!
		}
		else {
			return error('TanhGateVulkan: a must be a Variable')
		}
	}
}

// GELUGateVulkan: d_gelu on GPU (tanh approximation), gelu_input = original x
pub struct GELUGateVulkan[T] {
pub:
	gelu_input &vtl.Tensor[T]        = unsafe { nil }
	params     storage.VulkanStorageParams
}

pub fn gelu_gate_vulkan[T](gelu_input &vtl.Tensor[T], params storage.VulkanStorageParams) &GELUGateVulkan[T] {
	return &GELUGateVulkan[T]{gelu_input: gelu_input, params: params}
}

pub fn (g &GELUGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	x := g.gelu_input
	n := x.size
	$if T is f32 {
		dev := g.params.device
		size := vulkan.DeviceSize(u64(n) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		vk_x := x.vulkan(g.params)!
		defer { vk_x.release() or {} }
		mut out_buf := dev.buffer(size)!
		defer { out_buf.release() }
		vulkan.d_gelu(dev, out_buf, vk_grad.data.data, vk_x.data.data)!
		mut raw := []u8{len: n * 4}
		raw = out_buf.store(mut raw)!
		mut vals := []T{len: n}
		for i in 0 .. n {
			unsafe { vals[i] = T(*(&f32(&raw[i * 4]))) }
		}
		r0 := vtl.from_array[T](vals, x.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		// CPU fallback: GELU derivative (tanh approximation)
		c := vtl.cast[T](0.044715)
		sqrt_2_over_pi := vtl.cast[T](0.7978845608028654)
		r0 := gradient.nmap([x], fn [c, sqrt_2_over_pi] [T](vals []T, _ []int) T {
			xv := vals[1]
			inner := sqrt_2_over_pi * (xv + c * xv * xv * xv)
			th := T(math.tanh(f64(inner)))
			sech2 := vtl.cast[T](1) - th * th
			dgelu := vtl.cast[T](0.5) * (vtl.cast[T](1) + th) +
				vtl.cast[T](0.5) * xv * sech2 * sqrt_2_over_pi *
				(vtl.cast[T](1) + vtl.cast[T](3) * c * xv * xv)
			return vals[0] * dgelu
		})!
		return [r0]
	}
}

pub fn (g &GELUGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('GELUVulkan', g, result, [a])!
		}
		else {
			return error('GELUGateVulkan: a must be a Variable')
		}
	}
}
