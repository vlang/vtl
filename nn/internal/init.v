module internal

import arrays
import math
import vtl

// FanMode lists the supported public values for this module.
pub enum FanMode {
	fan_avg
	fan_in
	fan_out
}

// Distribution lists the supported public values for this module.
pub enum Distribution {
	uniform
	normal
}

// compute_fans exposes this operation as part of the public API.
pub fn compute_fans(shape []int) (int, int) {
	f0 := shape[0]
	f1 := shape[1]

	if shape.len == 1 {
		return f0, f1
	}

	product := arrays.fold(shape[1..], 1, fn (prev int, curr int) int {
		return prev * curr
	})
	return f0 * product, f1 * product
}

// variance_scaled exposes this operation as part of the public API.
pub fn variance_scaled[T](shape []int, scale T, fan_mode FanMode, distribution Distribution) &vtl.Tensor[T] {
	f0, f1 := compute_fans(shape)

	std := match fan_mode {
		.fan_in {
			math.sqrt(f64(scale) / f64(f0))
		}
		.fan_out {
			math.sqrt(f64(scale) / f64(f1))
		}
		.fan_avg {
			math.sqrt(f64(scale * vtl.cast[T](2)) / f64(f0 + f1))
		}
	}

	match distribution {
		.uniform {
			limit := vtl.cast[T](math.sqrt(3.0) * std)
			return vtl.random[T](-limit, limit, shape)
		}
		.normal {
			return vtl.normal[T](shape, mu: 0.0, sigma: std)
		}
	}
}

// kaiming_uniform exposes this operation as part of the public API.
pub fn kaiming_uniform[T](shape []int) &vtl.Tensor[T] {
	return variance_scaled(shape, vtl.cast[T](2), .fan_in, .uniform)
}

// kaiming_normal exposes this operation as part of the public API.
pub fn kaiming_normal[T](shape []int) &vtl.Tensor[T] {
	return variance_scaled(shape, vtl.cast[T](2), .fan_in, .normal)
}
