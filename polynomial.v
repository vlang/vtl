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
				root := -b / (2.0 * a)
				return [root, root]
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

// Chebyshev stores coefficients in ascending first-kind basis order:
// coefficients[i] multiplies T_i after mapping domain to window.
pub struct Chebyshev {
pub:
	coefficients []f64
	domain       [2]f64
	window       [2]f64
}

pub fn chebyshev(coefficients []f64, domain [2]f64, window [2]f64) !Chebyshev {
	validate_polynomial_mapping(domain, window, 'chebyshev')!
	if coefficients.len == 0 {
		return error('chebyshev requires at least one coefficient')
	}
	mut trimmed := coefficients.clone()
	for trimmed.len > 1 && trimmed[trimmed.len - 1] == 0 {
		trimmed.delete_last()
	}
	return Chebyshev{
		coefficients: trimmed
		domain:       domain
		window:       window
	}
}

pub fn (p Chebyshev) degree() int {
	return p.coefficients.len - 1
}

pub fn (p Chebyshev) eval(x f64) f64 {
	mapped_x := p.window[0] + (x - p.domain[0]) * (p.window[1] - p.window[0]) /
		(p.domain[1] - p.domain[0])
	mut next := 0.0
	mut next_next := 0.0
	for index := p.coefficients.len - 1; index >= 1; index-- {
		current := 2.0 * mapped_x * next - next_next + p.coefficients[index]
		next_next = next
		next = current
	}
	return mapped_x * next - next_next + p.coefficients[0]
}

pub fn (p Chebyshev) eval_tensor(input &Tensor[f64]) !&Tensor[f64] {
	return input.nmap([input], fn [p] (values []f64, _ []int) f64 {
		return p.eval(values[0])
	})
}

pub fn (p Chebyshev) to_polynomial() !Polynomial {
	validate_polynomial_mapping(p.domain, p.window, 'chebyshev')!
	if p.coefficients.len == 0 {
		return error('chebyshev requires at least one coefficient')
	}
	offset := p.window[0] - p.domain[0] * (p.window[1] - p.window[0]) /
		(p.domain[1] - p.domain[0])
	scale := (p.window[1] - p.window[0]) / (p.domain[1] - p.domain[0])
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('chebyshev: domain mapping overflowed during power-basis conversion')
	}
	mut previous := Polynomial{ coefficients: [1.0] }
	mut result := scale_polynomial(previous, p.coefficients[0])
	if p.coefficients.len == 1 {
		for coefficient in result.coefficients {
			if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
				return error('chebyshev: coefficients overflowed during power-basis conversion')
			}
		}
		return polynomial(result.coefficients)
	}
	mut current := Polynomial{ coefficients: [offset, scale] }
	result = result.add(scale_polynomial(current, p.coefficients[1]))
	for index in 2 .. p.coefficients.len {
		next := polynomial_multiply_affine(current, 2.0 * scale, 2.0 * offset).add(scale_polynomial(previous,
			-1.0))
		result = result.add(scale_polynomial(next, p.coefficients[index]))
		previous = current
		current = next
	}
	for coefficient in result.coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('chebyshev: coefficients overflowed during power-basis conversion')
		}
	}
	return polynomial(result.coefficients)
}

pub fn (p Chebyshev) derivative() !Chebyshev {
	ordinary := p.to_polynomial()!.derivative()
	return chebyshev_from_polynomial(ordinary, p.domain, p.window)
}

pub fn (p Chebyshev) integral(constant f64) !Chebyshev {
	ordinary := p.to_polynomial()!.integral(constant)
	return chebyshev_from_polynomial(ordinary, p.domain, p.window)
}

pub fn (p Chebyshev) roots() ![]f64 {
	return p.to_polynomial()!.roots()
}

pub fn (p Chebyshev) add(other Chebyshev) !Chebyshev {
	if p.domain != other.domain || p.window != other.window {
		return error('chebyshev addition requires matching domain and window')
	}
	size := if p.coefficients.len > other.coefficients.len {
		p.coefficients.len
	} else {
		other.coefficients.len
	}
	mut coefficients := []f64{len: size}
	for index in 0 .. size {
		if index < p.coefficients.len {
			coefficients[index] += p.coefficients[index]
		}
		if index < other.coefficients.len {
			coefficients[index] += other.coefficients[index]
		}
	}
	return chebyshev(coefficients, p.domain, p.window)
}

pub fn (p Chebyshev) mul(other Chebyshev) !Chebyshev {
	if p.domain != other.domain || p.window != other.window {
		return error('chebyshev multiplication requires matching domain and window')
	}
	left := p.to_polynomial()!
	right := other.to_polynomial()!
	return chebyshev_from_polynomial(left.mul(right), p.domain, p.window)
}

pub fn chebyshev_from_polynomial(p Polynomial, domain [2]f64, window [2]f64) !Chebyshev {
	validate_polynomial_mapping(domain, window, 'chebyshev_from_polynomial')!
	if p.coefficients.len == 0 {
		return error('chebyshev_from_polynomial requires at least one coefficient')
	}
	scale := (domain[1] - domain[0]) / (window[1] - window[0])
	offset := domain[0] - window[0] * scale
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('chebyshev_from_polynomial: domain mapping overflowed during conversion')
	}
	normalized_polynomial := polynomial(p.coefficients)!
	polynomial_in_window := polynomial_compose_affine(normalized_polynomial, scale, offset)
	degree := polynomial_in_window.degree()
	mut basis := []Polynomial{len: degree + 1}
	basis[0] = Polynomial{ coefficients: [1.0] }
	if degree > 0 {
		basis[1] = Polynomial{ coefficients: [0.0, 1.0] }
	}
	for index in 2 .. degree + 1 {
		basis[index] = polynomial_multiply_affine(basis[index - 1], 2.0, 0.0).add(scale_polynomial(basis[index - 2], -1.0))
	}
	mut residual := polynomial_in_window.coefficients.clone()
	mut coefficients := []f64{len: degree + 1}
	for index := degree; index >= 0; index-- {
		leading := basis[index].coefficients[index]
		coefficients[index] = residual[index] / leading
		for power in 0 .. index + 1 {
			residual[power] -= coefficients[index] * basis[index].coefficients[power]
		}
	}
	for coefficient in coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('chebyshev_from_polynomial: coefficients overflowed during conversion')
		}
	}
	return chebyshev(coefficients, domain, window)
}

pub fn chebyshev_fit(x []f64, y []f64, degree int, domain [2]f64, window [2]f64) !Chebyshev {
	validate_polynomial_mapping(domain, window, 'chebyshev_fit')!
	mut mapped_x := []f64{len: x.len}
	for index, value in x {
		mapped_x[index] = window[0] + (value - domain[0]) * (window[1] - window[0]) /
			(domain[1] - domain[0])
	}
	fitted := polynomial_fit(mapped_x, y, degree)!
	in_window_basis := chebyshev_from_polynomial(fitted, window, window)!
	return chebyshev(in_window_basis.coefficients, domain, window)
}

// Legendre stores coefficients in ascending basis order:
// coefficients[i] multiplies P_i after mapping domain to window.
pub struct Legendre {
pub:
	coefficients []f64
	domain       [2]f64
	window       [2]f64
}

pub fn legendre(coefficients []f64, domain [2]f64, window [2]f64) !Legendre {
	validate_polynomial_mapping(domain, window, 'legendre')!
	if coefficients.len == 0 {
		return error('legendre requires at least one coefficient')
	}
	mut trimmed := coefficients.clone()
	for trimmed.len > 1 && trimmed[trimmed.len - 1] == 0 {
		trimmed.delete_last()
	}
	return Legendre{
		coefficients: trimmed
		domain:       domain
		window:       window
	}
}

pub fn (p Legendre) degree() int {
	return p.coefficients.len - 1
}

pub fn (p Legendre) eval(x f64) f64 {
	mapped_x := p.window[0] + (x - p.domain[0]) * (p.window[1] - p.window[0]) /
		(p.domain[1] - p.domain[0])
	if p.coefficients.len == 1 {
		return p.coefficients[0]
	}
	mut previous := 1.0
	mut current := mapped_x
	mut result := p.coefficients[0] + p.coefficients[1] * current
	for index in 2 .. p.coefficients.len {
		next := ((2.0 * f64(index) - 1.0) * mapped_x * current -
			(f64(index) - 1.0) * previous) / f64(index)
		result += p.coefficients[index] * next
		previous = current
		current = next
	}
	return result
}

pub fn (p Legendre) eval_tensor(input &Tensor[f64]) !&Tensor[f64] {
	return input.nmap([input], fn [p] (values []f64, _ []int) f64 {
		return p.eval(values[0])
	})
}

pub fn (p Legendre) to_polynomial() !Polynomial {
	validate_polynomial_mapping(p.domain, p.window, 'legendre')!
	if p.coefficients.len == 0 {
		return error('legendre requires at least one coefficient')
	}
	offset := p.window[0] - p.domain[0] * (p.window[1] - p.window[0]) /
		(p.domain[1] - p.domain[0])
	scale := (p.window[1] - p.window[0]) / (p.domain[1] - p.domain[0])
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('legendre: domain mapping overflowed during power-basis conversion')
	}
	mut previous := Polynomial{ coefficients: [1.0] }
	mut result := scale_polynomial(previous, p.coefficients[0])
	if p.coefficients.len == 1 {
		if math.is_nan(result.coefficients[0]) || math.is_inf(result.coefficients[0], 0) {
			return error('legendre: coefficients overflowed during power-basis conversion')
		}
		return polynomial(result.coefficients)
	}
	mut current := Polynomial{ coefficients: [offset, scale] }
	result = result.add(scale_polynomial(current, p.coefficients[1]))
	for index in 2 .. p.coefficients.len {
		first_factor := (2.0 * f64(index) - 1.0) / f64(index)
		second_factor := (f64(index) - 1.0) / f64(index)
		next := polynomial_multiply_affine(current, first_factor * scale,
			first_factor * offset).add(scale_polynomial(previous, -second_factor))
		result = result.add(scale_polynomial(next, p.coefficients[index]))
		previous = current
		current = next
	}
	for coefficient in result.coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('legendre: coefficients overflowed during power-basis conversion')
		}
	}
	return polynomial(result.coefficients)
}

pub fn (p Legendre) derivative() !Legendre {
	ordinary := p.to_polynomial()!.derivative()
	return legendre_from_polynomial(ordinary, p.domain, p.window)
}

pub fn (p Legendre) integral(constant f64) !Legendre {
	ordinary := p.to_polynomial()!.integral(constant)
	return legendre_from_polynomial(ordinary, p.domain, p.window)
}

pub fn (p Legendre) roots() ![]f64 {
	return p.to_polynomial()!.roots()
}

pub fn (p Legendre) add(other Legendre) !Legendre {
	if p.domain != other.domain || p.window != other.window {
		return error('legendre addition requires matching domain and window')
	}
	size := if p.coefficients.len > other.coefficients.len {
		p.coefficients.len
	} else {
		other.coefficients.len
	}
	mut coefficients := []f64{len: size}
	for index in 0 .. size {
		if index < p.coefficients.len {
			coefficients[index] += p.coefficients[index]
		}
		if index < other.coefficients.len {
			coefficients[index] += other.coefficients[index]
		}
	}
	return legendre(coefficients, p.domain, p.window)
}

pub fn (p Legendre) mul(other Legendre) !Legendre {
	if p.domain != other.domain || p.window != other.window {
		return error('legendre multiplication requires matching domain and window')
	}
	left := p.to_polynomial()!
	right := other.to_polynomial()!
	return legendre_from_polynomial(left.mul(right), p.domain, p.window)
}

pub fn legendre_from_polynomial(p Polynomial, domain [2]f64, window [2]f64) !Legendre {
	validate_polynomial_mapping(domain, window, 'legendre_from_polynomial')!
	if p.coefficients.len == 0 {
		return error('legendre_from_polynomial requires at least one coefficient')
	}
	scale := (domain[1] - domain[0]) / (window[1] - window[0])
	offset := domain[0] - window[0] * scale
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('legendre_from_polynomial: domain mapping overflowed during conversion')
	}
	normalized_polynomial := polynomial(p.coefficients)!
	polynomial_in_window := polynomial_compose_affine(normalized_polynomial, scale, offset)
	degree := polynomial_in_window.degree()
	mut basis := []Polynomial{len: degree + 1}
	basis[0] = Polynomial{ coefficients: [1.0] }
	if degree > 0 {
		basis[1] = Polynomial{ coefficients: [0.0, 1.0] }
	}
	for index in 2 .. degree + 1 {
		first_factor := (2.0 * f64(index) - 1.0) / f64(index)
		second_factor := (f64(index) - 1.0) / f64(index)
		basis[index] = polynomial_multiply_affine(basis[index - 1], first_factor,
			0.0).add(scale_polynomial(basis[index - 2], -second_factor))
	}
	mut residual := polynomial_in_window.coefficients.clone()
	mut coefficients := []f64{len: degree + 1}
	for index := degree; index >= 0; index-- {
		leading := basis[index].coefficients[index]
		coefficients[index] = residual[index] / leading
		for power in 0 .. index + 1 {
			residual[power] -= coefficients[index] * basis[index].coefficients[power]
		}
	}
	for coefficient in coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('legendre_from_polynomial: coefficients overflowed during conversion')
		}
	}
	return legendre(coefficients, domain, window)
}

pub fn legendre_fit(x []f64, y []f64, degree int, domain [2]f64, window [2]f64) !Legendre {
	validate_polynomial_mapping(domain, window, 'legendre_fit')!
	mut mapped_x := []f64{len: x.len}
	for index, value in x {
		mapped_x[index] = window[0] + (value - domain[0]) * (window[1] - window[0]) /
			(domain[1] - domain[0])
	}
	fitted := polynomial_fit(mapped_x, y, degree)!
	in_window_basis := legendre_from_polynomial(fitted, window, window)!
	return legendre(in_window_basis.coefficients, domain, window)
}

enum HermiteFamily {
	physicists
	probabilists
}

// Hermite stores physicists' Hermite coefficients in ascending basis order.
pub struct Hermite {
pub:
	coefficients []f64
	domain       [2]f64
	window       [2]f64
}

pub fn hermite(coefficients []f64, domain [2]f64, window [2]f64) !Hermite {
	validate_polynomial_mapping(domain, window, 'hermite')!
	return Hermite{
		coefficients: new_orthogonal_coefficients(coefficients, 'hermite')!
		domain:       domain
		window:       window
	}
}

pub fn (p Hermite) degree() int {
	return p.coefficients.len - 1
}

pub fn (p Hermite) eval(x f64) f64 {
	mapped_x := map_polynomial_domain(x, p.domain, p.window)
	return hermite_series_eval(p.coefficients, mapped_x, .physicists)
}

pub fn (p Hermite) eval_tensor(input &Tensor[f64]) !&Tensor[f64] {
	return input.nmap([input], fn [p] (values []f64, _ []int) f64 {
		return p.eval(values[0])
	})
}

pub fn (p Hermite) to_polynomial() !Polynomial {
	return orthogonal_series_to_polynomial(p.coefficients, p.domain, p.window, .physicists,
		'hermite')
}

pub fn (p Hermite) derivative() !Hermite {
	ordinary := p.to_polynomial()!.derivative()
	coefficients := orthogonal_series_from_polynomial(ordinary, p.domain, p.window, .physicists,
		'hermite_from_polynomial')!
	return hermite(coefficients, p.domain, p.window)
}

pub fn (p Hermite) integral(constant f64) !Hermite {
	ordinary := p.to_polynomial()!.integral(constant)
	coefficients := orthogonal_series_from_polynomial(ordinary, p.domain, p.window, .physicists,
		'hermite_from_polynomial')!
	return hermite(coefficients, p.domain, p.window)
}

pub fn (p Hermite) roots() ![]f64 {
	return p.to_polynomial()!.roots()
}

pub fn (p Hermite) add(other Hermite) !Hermite {
	coefficients := add_orthogonal_coefficients(p.coefficients, other.coefficients, p.domain,
		p.window, other.domain, other.window, 'hermite')!
	return hermite(coefficients, p.domain, p.window)
}

pub fn (p Hermite) mul(other Hermite) !Hermite {
	if p.domain != other.domain || p.window != other.window {
		return error('hermite multiplication requires matching domain and window')
	}
	product := p.to_polynomial()!.mul(other.to_polynomial()!)
	coefficients := orthogonal_series_from_polynomial(product, p.domain, p.window, .physicists,
		'hermite_from_polynomial')!
	return hermite(coefficients, p.domain, p.window)
}

pub fn hermite_from_polynomial(p Polynomial, domain [2]f64, window [2]f64) !Hermite {
	coefficients := orthogonal_series_from_polynomial(p, domain, window, .physicists,
		'hermite_from_polynomial')!
	return hermite(coefficients, domain, window)
}

pub fn hermite_fit(x []f64, y []f64, degree int, domain [2]f64, window [2]f64) !Hermite {
	coefficients := orthogonal_series_fit(x, y, degree, domain, window, .physicists, 'hermite_fit')!
	return hermite(coefficients, domain, window)
}

// HermiteE stores probabilists' Hermite coefficients in ascending basis order.
pub struct HermiteE {
pub:
	coefficients []f64
	domain       [2]f64
	window       [2]f64
}

pub fn hermite_e(coefficients []f64, domain [2]f64, window [2]f64) !HermiteE {
	validate_polynomial_mapping(domain, window, 'hermite_e')!
	return HermiteE{
		coefficients: new_orthogonal_coefficients(coefficients, 'hermite_e')!
		domain:       domain
		window:       window
	}
}

pub fn (p HermiteE) degree() int {
	return p.coefficients.len - 1
}

pub fn (p HermiteE) eval(x f64) f64 {
	mapped_x := map_polynomial_domain(x, p.domain, p.window)
	return hermite_series_eval(p.coefficients, mapped_x, .probabilists)
}

pub fn (p HermiteE) eval_tensor(input &Tensor[f64]) !&Tensor[f64] {
	return input.nmap([input], fn [p] (values []f64, _ []int) f64 {
		return p.eval(values[0])
	})
}

pub fn (p HermiteE) to_polynomial() !Polynomial {
	return orthogonal_series_to_polynomial(p.coefficients, p.domain, p.window, .probabilists,
		'hermite_e')
}

pub fn (p HermiteE) derivative() !HermiteE {
	ordinary := p.to_polynomial()!.derivative()
	coefficients := orthogonal_series_from_polynomial(ordinary, p.domain, p.window, .probabilists,
		'hermite_e_from_polynomial')!
	return hermite_e(coefficients, p.domain, p.window)
}

pub fn (p HermiteE) integral(constant f64) !HermiteE {
	ordinary := p.to_polynomial()!.integral(constant)
	coefficients := orthogonal_series_from_polynomial(ordinary, p.domain, p.window, .probabilists,
		'hermite_e_from_polynomial')!
	return hermite_e(coefficients, p.domain, p.window)
}

pub fn (p HermiteE) roots() ![]f64 {
	return p.to_polynomial()!.roots()
}

pub fn (p HermiteE) add(other HermiteE) !HermiteE {
	coefficients := add_orthogonal_coefficients(p.coefficients, other.coefficients, p.domain,
		p.window, other.domain, other.window, 'hermite_e')!
	return hermite_e(coefficients, p.domain, p.window)
}

pub fn (p HermiteE) mul(other HermiteE) !HermiteE {
	if p.domain != other.domain || p.window != other.window {
		return error('hermite_e multiplication requires matching domain and window')
	}
	product := p.to_polynomial()!.mul(other.to_polynomial()!)
	coefficients := orthogonal_series_from_polynomial(product, p.domain, p.window, .probabilists,
		'hermite_e_from_polynomial')!
	return hermite_e(coefficients, p.domain, p.window)
}

pub fn hermite_e_from_polynomial(p Polynomial, domain [2]f64, window [2]f64) !HermiteE {
	coefficients := orthogonal_series_from_polynomial(p, domain, window, .probabilists,
		'hermite_e_from_polynomial')!
	return hermite_e(coefficients, domain, window)
}

pub fn hermite_e_fit(x []f64, y []f64, degree int, domain [2]f64, window [2]f64) !HermiteE {
	coefficients := orthogonal_series_fit(x, y, degree, domain, window, .probabilists,
		'hermite_e_fit')!
	return hermite_e(coefficients, domain, window)
}

fn new_orthogonal_coefficients(coefficients []f64, family string) ![]f64 {
	if coefficients.len == 0 {
		return error('${family} requires at least one coefficient')
	}
	mut trimmed := coefficients.clone()
	for trimmed.len > 1 && trimmed[trimmed.len - 1] == 0 {
		trimmed.delete_last()
	}
	return trimmed
}

fn map_polynomial_domain(x f64, domain [2]f64, window [2]f64) f64 {
	return window[0] + (x - domain[0]) * (window[1] - window[0]) / (domain[1] - domain[0])
}

fn hermite_series_eval(coefficients []f64, x f64, family HermiteFamily) f64 {
	if coefficients.len == 1 {
		return coefficients[0]
	}
	mut previous := 1.0
	mut current := if family == .physicists { 2.0 * x } else { x }
	mut result := coefficients[0] + coefficients[1] * current
	for index in 2 .. coefficients.len {
		linear_factor := if family == .physicists { 2.0 } else { 1.0 }
		lower_factor := if family == .physicists {
			2.0 * f64(index - 1)
		} else {
			f64(index - 1)
		}
		next := linear_factor * x * current - lower_factor * previous
		result += coefficients[index] * next
		previous = current
		current = next
	}
	return result
}

fn orthogonal_series_to_polynomial(coefficients []f64, domain [2]f64, window [2]f64, family HermiteFamily, name string) !Polynomial {
	validate_polynomial_mapping(domain, window, name)!
	if coefficients.len == 0 {
		return error('${name}: coefficients must not be empty')
	}
	scale := (window[1] - window[0]) / (domain[1] - domain[0])
	offset := window[0] - domain[0] * scale
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('${name}: domain mapping overflowed during conversion')
	}
	mut previous := Polynomial{ coefficients: [1.0] }
	mut result := scale_polynomial(previous, coefficients[0])
	if coefficients.len > 1 {
		linear_factor := if family == .physicists { 2.0 } else { 1.0 }
		mut current := Polynomial{
			coefficients: [linear_factor * offset, linear_factor * scale]
		}
		result = result.add(scale_polynomial(current, coefficients[1]))
		for index in 2 .. coefficients.len {
			lower_factor := if family == .physicists {
				2.0 * f64(index - 1)
			} else {
				f64(index - 1)
			}
			next := polynomial_multiply_affine(current, linear_factor * scale,
				linear_factor * offset).add(scale_polynomial(previous, -lower_factor))
			result = result.add(scale_polynomial(next, coefficients[index]))
			previous = current
			current = next
		}
	}
	for coefficient in result.coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('${name}: coefficients overflowed during power-basis conversion')
		}
	}
	return polynomial(result.coefficients)
}

fn orthogonal_series_from_polynomial(p Polynomial, domain [2]f64, window [2]f64, family HermiteFamily, name string) ![]f64 {
	validate_polynomial_mapping(domain, window, name)!
	if p.coefficients.len == 0 {
		return error('${name}: polynomial coefficients must not be empty')
	}
	scale := (domain[1] - domain[0]) / (window[1] - window[0])
	offset := domain[0] - window[0] * scale
	if math.is_nan(offset) || math.is_inf(offset, 0) || math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('${name}: domain mapping overflowed during conversion')
	}
	normalized_polynomial := polynomial(p.coefficients)!
	polynomial_in_window := polynomial_compose_affine(normalized_polynomial, scale, offset)
	degree := polynomial_in_window.degree()
	mut basis := []Polynomial{len: degree + 1}
	basis[0] = Polynomial{ coefficients: [1.0] }
	if degree > 0 {
		linear_factor := if family == .physicists { 2.0 } else { 1.0 }
		basis[1] = Polynomial{ coefficients: [0.0, linear_factor] }
	}
	for index in 2 .. degree + 1 {
		linear_factor := if family == .physicists { 2.0 } else { 1.0 }
		lower_factor := if family == .physicists {
			2.0 * f64(index - 1)
		} else {
			f64(index - 1)
		}
		basis[index] = polynomial_multiply_affine(basis[index - 1], linear_factor,
			0.0).add(scale_polynomial(basis[index - 2], -lower_factor))
	}
	mut residual := polynomial_in_window.coefficients.clone()
	mut coefficients := []f64{len: degree + 1}
	for index := degree; index >= 0; index-- {
		leading := basis[index].coefficients[index]
		coefficients[index] = residual[index] / leading
		for power in 0 .. index + 1 {
			residual[power] -= coefficients[index] * basis[index].coefficients[power]
		}
	}
	for coefficient in coefficients {
		if math.is_nan(coefficient) || math.is_inf(coefficient, 0) {
			return error('${name}: coefficients overflowed during conversion')
		}
	}
	return coefficients
}

fn orthogonal_series_fit(x []f64, y []f64, degree int, domain [2]f64, window [2]f64, family HermiteFamily, name string) ![]f64 {
	validate_polynomial_mapping(domain, window, name)!
	mut mapped_x := []f64{len: x.len}
	for index, value in x {
		mapped_x[index] = map_polynomial_domain(value, domain, window)
	}
	fitted := polynomial_fit(mapped_x, y, degree)!
	return orthogonal_series_from_polynomial(fitted, window, window, family, name)
}

fn add_orthogonal_coefficients(left []f64, right []f64, left_domain [2]f64, left_window [2]f64, right_domain [2]f64, right_window [2]f64, name string) ![]f64 {
	if left_domain != right_domain || left_window != right_window {
		return error('${name} addition requires matching domain and window')
	}
	size := if left.len > right.len { left.len } else { right.len }
	mut coefficients := []f64{len: size}
	for index in 0 .. size {
		if index < left.len {
			coefficients[index] += left[index]
		}
		if index < right.len {
			coefficients[index] += right[index]
		}
	}
	return coefficients
}

fn validate_polynomial_mapping(domain [2]f64, window [2]f64, operation string) ! {
	for value in [domain[0], domain[1], window[0], window[1]] {
		if math.is_nan(value) || math.is_inf(value, 0) {
			return error('${operation}: domain and window bounds must be finite')
		}
	}
	domain_width := domain[1] - domain[0]
	window_width := window[1] - window[0]
	if domain_width == 0 || window_width == 0 || math.is_inf(domain_width, 0)
		|| math.is_inf(window_width, 0) {
		return error('${operation}: domain and window widths must be finite and non-zero')
	}
}

fn polynomial_multiply_affine(p Polynomial, linear f64, constant f64) Polynomial {
	mut coefficients := []f64{len: p.coefficients.len + 1}
	for index, value in p.coefficients {
		coefficients[index] += value * constant
		coefficients[index + 1] += value * linear
	}
	return Polynomial{ coefficients: coefficients }
}

fn polynomial_compose_affine(p Polynomial, linear f64, constant f64) Polynomial {
	mut result := Polynomial{ coefficients: [p.coefficients[p.coefficients.len - 1]] }
	for index := p.coefficients.len - 2; index >= 0; index-- {
		result = polynomial_multiply_affine(result, linear, constant)
		mut coefficients := result.coefficients.clone()
		coefficients[0] += p.coefficients[index]
		result = Polynomial{ coefficients: coefficients }
	}
	return result
}

fn scale_polynomial(p Polynomial, factor f64) Polynomial {
	mut coefficients := p.coefficients.clone()
	for index in 0 .. coefficients.len {
		coefficients[index] *= factor
	}
	return Polynomial{ coefficients: coefficients }
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
