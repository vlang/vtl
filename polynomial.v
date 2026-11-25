module vtl

import math

// Polynomial stores power-series coefficients in ascending order:
// coefficients[i] is the coefficient of x^i.
pub struct Polynomial {
pub:
	coefficients []f64
}

pub fn polynomial(coefficients []f64) !Polynomial {
	if coefficients.len == 0 {
		return error('polynomial requires at least one coefficient')
	}
	mut result := Polynomial{ coefficients: coefficients.clone() }
	result.trim()
	return result
}

fn (mut p Polynomial) trim() {
	for p.coefficients.len > 1 && p.coefficients[p.coefficients.len - 1] == 0.0 {
		p.coefficients.delete_last()
	}
}

pub fn (p Polynomial) degree() int {
	return p.coefficients.len - 1
}

pub fn (p Polynomial) eval(x f64) f64 {
	mut result := 0.0
	for i := p.coefficients.len - 1; i >= 0; i-- {
		result = result * x + p.coefficients[i]
	}
	return result
}

pub fn (p Polynomial) eval_tensor(input &Tensor[f64]) !&Tensor[f64] {
	return input.nmap([input], fn [p] (values []f64, _ []int) f64 {
		return p.eval(values[0])
	})
}

pub fn (p Polynomial) add(other Polynomial) Polynomial {
	size := if p.coefficients.len > other.coefficients.len {
		p.coefficients.len
	} else {
		other.coefficients.len
	}
	mut coefficients := []f64{len: size}
	for i in 0 .. size {
		if i < p.coefficients.len {
			coefficients[i] += p.coefficients[i]
		}
		if i < other.coefficients.len {
			coefficients[i] += other.coefficients[i]
		}
	}
	mut result := Polynomial{ coefficients: coefficients }
	result.trim()
	return result
}

pub fn (p Polynomial) mul(other Polynomial) Polynomial {
	mut coefficients := []f64{len: p.coefficients.len + other.coefficients.len - 1}
	for i, a in p.coefficients {
		for j, b in other.coefficients {
			coefficients[i + j] += a * b
		}
	}
	return Polynomial{ coefficients: coefficients }
}

pub fn (p Polynomial) derivative() Polynomial {
	if p.coefficients.len == 1 {
		return Polynomial{ coefficients: [0.0] }
	}
	mut coefficients := []f64{len: p.coefficients.len - 1}
	for i in 1 .. p.coefficients.len {
		coefficients[i - 1] = f64(i) * p.coefficients[i]
	}
	return Polynomial{ coefficients: coefficients }
}

pub fn (p Polynomial) integral(constant f64) Polynomial {
	mut coefficients := []f64{len: p.coefficients.len + 1}
	coefficients[0] = constant
	for i, coefficient in p.coefficients {
		coefficients[i + 1] = coefficient / f64(i + 1)
	}
	return Polynomial{ coefficients: coefficients }
}

pub fn (p Polynomial) roots() ![]f64 {
	match p.degree() {
		0 { return error('constant polynomial has no finite roots') }
		1 { return [-p.coefficients[0] / p.coefficients[1]] }
		2 {
			a, b, c := p.coefficients[2], p.coefficients[1], p.coefficients[0]
			discriminant := b * b - 4.0 * a * c
			if discriminant < 0.0 {
				return []f64{}
			}
			if discriminant == 0.0 {
				return [-b / (2.0 * a)]
			}
			root := math.sqrt(discriminant)
			// Use the product of roots for the second root to avoid cancellation.
			if b >= 0 {
				q := -0.5 * (b + root)
				return [q / a, c / q]
			}
			q := -0.5 * (b - root)
			return [c / q, q / a]
		}
		else { return error('roots currently supports degree up to two') }
	}
}

// polynomial_fit computes a least-squares power-series fit with a scaled
// Vandermonde matrix and Householder QR.
pub fn polynomial_fit(x []f64, y []f64, degree int) !Polynomial {
	if x.len != y.len || x.len == 0 {
		return error('polynomial_fit requires equally sized non-empty samples')
	}
	if degree < 0 || degree >= x.len {
		return error('polynomial_fit degree must be non-negative and smaller than sample count')
	}
	mut min_x := x[0]
	mut max_x := x[0]
	for value in x {
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('polynomial_fit requires finite x values')
		}
		min_x = math.min(min_x, value)
		max_x = math.max(max_x, value)
	}
	for value in y {
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('polynomial_fit requires finite y values')
		}
	}
	center := min_x / 2.0 + max_x / 2.0
	mut scale := math.max(math.abs(min_x - center), math.abs(max_x - center))
	if scale == 0 {
		scale = 1.0
	}
	size := degree + 1
	mut matrix := [][]f64{len: x.len}
	for row, value in x {
		matrix[row] = []f64{len: size}
		normalized_x := (value - center) / scale
		mut monomial := 1.0
		for column in 0 .. size {
			matrix[row][column] = monomial
			monomial *= normalized_x
		}
	}
	mut transformed_y := y.clone()
	for pivot in 0 .. size {
		mut norm_squared := 0.0
		for row in pivot .. x.len {
			norm_squared += matrix[row][pivot] * matrix[row][pivot]
		}
		norm := math.sqrt(norm_squared)
		if norm <= 1e-14 {
			return error('polynomial_fit samples produce a rank-deficient system')
		}
		alpha := if matrix[pivot][pivot] >= 0 {
			-norm
		} else {
			norm
		}
		mut reflector := []f64{len: x.len - pivot}
		reflector[0] = matrix[pivot][pivot] - alpha
		for row in pivot + 1 .. x.len {
			reflector[row - pivot] = matrix[row][pivot]
		}
		mut reflector_norm_squared := 0.0
		for value in reflector {
			reflector_norm_squared += value * value
		}
		for column in pivot + 1 .. size {
			mut dot := 0.0
			for row in pivot .. x.len {
				dot += reflector[row - pivot] * matrix[row][column]
			}
			factor := 2.0 * dot / reflector_norm_squared
			for row in pivot .. x.len {
				matrix[row][column] -= factor * reflector[row - pivot]
			}
		}
		mut dot_y := 0.0
		for row in pivot .. x.len {
			dot_y += reflector[row - pivot] * transformed_y[row]
		}
		factor_y := 2.0 * dot_y / reflector_norm_squared
		for row in pivot .. x.len {
			transformed_y[row] -= factor_y * reflector[row - pivot]
		}
		matrix[pivot][pivot] = alpha
		for row in pivot + 1 .. x.len {
			matrix[row][pivot] = 0.0
		}
	}
	mut normalized_coefficients := []f64{len: size}
	for row := size - 1; row >= 0; row-- {
		normalized_coefficients[row] = transformed_y[row]
		for column in row + 1 .. size {
			normalized_coefficients[row] -= matrix[row][column] * normalized_coefficients[column]
		}
		normalized_coefficients[row] /= matrix[row][row]
	}

	inverse_scale := 1.0 / scale
	offset := -center / scale
	mut coefficients := [normalized_coefficients[size - 1]]
	for index := size - 2; index >= 0; index-- {
		mut expanded := []f64{len: coefficients.len + 1}
		for power, coefficient in coefficients {
			expanded[power] += coefficient * offset
			expanded[power + 1] += coefficient * inverse_scale
		}
		expanded[0] += normalized_coefficients[index]
		coefficients = expanded.clone()
	}
	for coefficient in coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('polynomial_fit coefficients overflowed during conversion to the power basis')
		}
	}
	return polynomial(coefficients)
}
