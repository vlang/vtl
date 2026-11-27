module main

import vtl

fn main() {
	x := [-1.0, 0.0, 1.0, 2.0]
	y := [2.0, 1.0, 2.0, 5.0]
	fitted := vtl.polynomial_fit(x, y, 2)!
	println('ascending coefficients: ${fitted.coefficients}')
	println('p(3): ${fitted.eval(3.0)}')
	domain := [f64(-1.0), 2.0]!
	window := [f64(-1.0), 1.0]!
	chebyshev_fit := vtl.chebyshev_fit(x, y, 2, domain, window)!
	println('ascending Chebyshev coefficients: ${chebyshev_fit.coefficients}')
	println('Chebyshev p(3): ${chebyshev_fit.eval(3.0)}')
	legendre_fit := vtl.legendre_fit(x, y, 2, domain, window)!
	println('ascending Legendre coefficients: ${legendre_fit.coefficients}')
	println('Legendre p(3): ${legendre_fit.eval(3.0)}')
	hermite_fit := vtl.hermite_fit(x, y, 2, domain, window)!
	println('ascending Hermite coefficients: ${hermite_fit.coefficients}')
	println('Hermite p(3): ${hermite_fit.eval(3.0)}')
	hermite_e_fit := vtl.hermite_e_fit(x, y, 2, domain, window)!
	println('ascending HermiteE coefficients: ${hermite_e_fit.coefficients}')
	println('HermiteE p(3): ${hermite_e_fit.eval(3.0)}')
}
