module tests

import vtl.array_api as xp

fn test_array_api_namespace_is_importable_by_consumers() {
	values := xp.arange[int](0.0, 4.0, 1.0)!
	assert xp.standard_version == '2024.12'
	assert values.to_array() == [0, 1, 2, 3]
	assert xp.supports('zeros')
}
