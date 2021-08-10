module storage

fn test_cpu_with_default() {
	c := new_cpu<f64>(2, 0, 1.0)
	array := c.to_array()
	assert array.len == 2
	assert array[0] == 1.0
}
