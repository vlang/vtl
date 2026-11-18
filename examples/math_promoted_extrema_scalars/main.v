import math
import vtl

fn main() {
	values := vtl.from_1d[f32]([f32(math.nan()), -0.0, 1.5])!
	maximum := vtl.maximum_scalar_promoted[f32, f32, f64](values, 0.25)!
	fmaximum := vtl.fmax_scalar_left_promoted[f32, f32, f64](0.25, values)!
	println(maximum.to_array())
	println(fmaximum.to_array())

	counts := vtl.from_1d[i8]([2, -3])!
	clamped := vtl.maximum_scalar_promoted[i8, i8, int](counts, 1)!
	println(clamped.to_array())
}
