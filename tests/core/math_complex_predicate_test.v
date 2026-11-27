module core

import math
import math.complex
import vtl

fn test_is_real_and_is_complex_match_numpy_complex_component_rules() ! {
	values := vtl.from_1d[complex.Complex]([
		complex.Complex{ re: 1.0 },
		complex.Complex{ re: 2.0, im: -0.0 },
		complex.Complex{ re: math.nan() },
		complex.Complex{ re: 1.0, im: math.nan() },
		complex.Complex{ im: 3.0 },
	])!

	assert values.is_real().to_array() == [true, true, true, false, false]
	assert values.is_complex().to_array() == [false, false, false, true, true]
	assert vtl.empty[complex.Complex]([0]).is_real().shape == [0]
	assert vtl.empty[complex.Complex]([0]).is_complex().shape == [0]
}

fn test_is_real_and_is_complex_classify_real_and_boolean_tensors() ! {
	real_values := vtl.from_1d[f64]([math.nan(), math.inf(1), -0.0])!
	assert real_values.is_real().to_array() == [true, true, true]
	assert real_values.is_complex().to_array() == [false, false, false]

	boolean_values := vtl.from_1d([true, false])!
	assert boolean_values.is_real().to_array() == [true, true]
	assert boolean_values.is_complex().to_array() == [false, false]
}
