module storage

fn test_cpu_with_default() {
	s := new_storage<f64>(2, 0, 1.0)
	array := s.to_array()
	assert array.len == 2
	assert array[0] == 1.0
}
