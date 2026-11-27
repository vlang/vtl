module main

import math
import vtl

fn main() {
	values := vtl.from_array[i16]([1, 7, -2, 4], [2, 2]) or { panic(err) }
	limits := vtl.from_1d[f32]([0.0, 5.0]) or { panic(err) }
	maximum := vtl.maximum_promoted[f32, i16, f32](values, limits) or { panic(err) }
	minimum := vtl.minimum_promoted[f32, i16, f32](values, limits) or { panic(err) }

	with_nan := vtl.from_1d[f32]([math.nan(), 4.0]) or { panic(err) }
	fallback := vtl.from_1d[i16]([2, 5]) or { panic(err) }
	finite_maximum := vtl.fmax_promoted[f32, f32, i16](with_nan, fallback) or { panic(err) }
	finite_minimum := vtl.fmin_promoted[f32, f32, i16](with_nan, fallback) or { panic(err) }

	println('maximum: ${maximum.to_array()}')
	println('minimum: ${minimum.to_array()}')
	println('fmax: ${finite_maximum.to_array()}')
	println('fmin: ${finite_minimum.to_array()}')
}
