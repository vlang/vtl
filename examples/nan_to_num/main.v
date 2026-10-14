module main

import vtl
import math
import math.complex as vcomplex

fn main() {
	values := vtl.from_1d([math.nan(), math.inf(1), math.inf(-1), 2.5])!
	cleaned := values.nan_to_num()
	println('Default finite replacements:')
	println(cleaned.to_array())
	custom := values.nan_to_num(nan: -1.0, posinf: 10.0, neginf: -10.0)
	println('Custom replacements:')
	println(custom.to_array())

	complex_values := vtl.from_1d([
		vcomplex.Complex{ re: math.inf(1), im: math.nan() },
		vcomplex.Complex{ re: math.inf(-1), im: 2.0 },
	])!
	cleaned_complex := complex_values.nan_to_num(nan: -1.0, posinf: 10.0, neginf: -10.0)
	println('Component-wise complex replacements:')
	println(cleaned_complex.to_array())
}
