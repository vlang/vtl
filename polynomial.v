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
			return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
		}
		else { return error('roots currently supports degree up to two') }
	}
}

// polynomial_fit computes a least-squares power-series fit using normal
// equations. It is intended for small, well-conditioned problems.
pub fn polynomial_fit(x []f64, y []f64, degree int) !Polynomial {
	if x.len != y.len || x.len == 0 {
		return error('polynomial_fit requires equally sized non-empty samples')
	}
	if degree < 0 || degree >= x.len {
		return error('polynomial_fit degree must be non-negative and smaller than sample count')
	}
	size := degree + 1
	mut matrix := [][]f64{len: size}
	for row in 0 .. size {
		matrix[row] = []f64{len: size + 1}
	}
	for row in 0 .. size {
		for column in 0 .. size {
			for _, value in x {
				matrix[row][column] += math.pow(value, f64(row + column))
			}
		}
		for sample, value in x {
			matrix[row][size] += math.pow(value, f64(row)) * y[sample]
		}
	}
	for pivot in 0 .. size {
		mut pivot_row := pivot
		for row in pivot + 1 .. size {
			if math.abs(matrix[row][pivot]) > math.abs(matrix[pivot_row][pivot]) {
				pivot_row = row
			}
		}
		if math.abs(matrix[pivot_row][pivot]) < 1e-14 {
			return error('polynomial_fit samples produce a singular system')
		}
		matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
		for row in pivot + 1 .. size {
			factor := matrix[row][pivot] / matrix[pivot][pivot]
			for column in pivot .. size + 1 {
				matrix[row][column] -= factor * matrix[pivot][column]
			}
		}
	}
	mut coefficients := []f64{len: size}
	for row := size - 1; row >= 0; row-- {
		coefficients[row] = matrix[row][size]
		for column in row + 1 .. size {
			coefficients[row] -= matrix[row][column] * coefficients[column]
		}
		coefficients[row] /= matrix[row][row]
	}
	return polynomial(coefficients)
}
