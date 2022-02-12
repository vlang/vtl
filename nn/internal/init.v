module internal

import math
import vtl

pub enum FanMode {
	fan_avg
	fan_in
	fan_out
}

pub enum Distribution {
	uniform
	normal
}

fn prod(prev int, curr int) int {
	return prev * curr
}

pub fn compute_fans(shape []int) (int, int) {
	f0 := shape[0]
	f1 := shape[1]

	if shape.len == 1 {
		return f0, f1
	}

	product := shape[1..].reduce(prod, 1)
	return f0 * product, f1 * product
}

pub fn variant_scaled<T>(shape []int, scale T, fan_mode FanMode, distribution Distribution) &vtl.Tensor<T> {
	f0, f1 := compute_fans(shape)

	std := match fan_mode {
		.fan_in {
			T(math.sqrt(f64(scale) / f64(f0)))
		}
		.fan_out {
			T(math.sqrt(f64(scale) / f64(f1)))
		}
		.fan_avg {
			T(math.sqrt(f64(scale * T(2)) / f64(f0 + f1)))
		}
	}

	match distribution {
		.uniform {
			limit := T(math.sqrt(3.0) * std)
			return vtl.random<T>(-limit, limit, shape)
		}
		.normal {
			panic('not implemented')
		}
	}
}

pub fn kaiming_uniform<T>(shape []int) &vtl.Tensor<T> {
	return variant_scaled(shape, T(2), .fan_in, .uniform)
}

pub fn kaiming_normal<T>(shape []int) &vtl.Tensor<T> {
	return variant_scaled(shape, T(2), .fan_in, .normal)
}
