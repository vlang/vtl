module storage

fn test_storage_with_default() {
	s := new_storage<f64>(2, 0, 1.0, .cpu)
	match s {
		CpuStorage<f64> {
			array := s.to_array()
			assert array.len == 2
			assert array[0] == 1.0
		}
		else {
			panic('This should not happen')
		}
	}
}
