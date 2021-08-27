module storage

fn test_storage_with_default() {
	s := new_storage<f64>(2, 0, 1.0, .cpu)
	match s {
		CpuStorage<f64> {
			array := s.to_array()
			assert array == [1.0, 1]
		}
		else {
			panic('This should not happen')
		}
	}
}

fn test_storage_set() {
	s := new_storage<f64>(2, 0, 1.0, .cpu)
	storage_set<f64>(s, 1, 2.0)
	match s {
		CpuStorage<f64> {
			array := s.to_array()
			assert array == [1.0, 2]
		}
		else {
			panic('This should not happen')
		}
	}
}

fn test_storage_fill() {
	s := new_storage<f64>(2, 0, 1.0, .cpu)
	storage_fill<f64>(s, 2.0)
	match s {
		CpuStorage<f64> {
			array := s.to_array()
			assert array == [2.0, 2]
		}
		else {
			panic('This should not happen')
		}
	}
}

fn test_storage_offset() {
	s := new_storage<f64>(5, 0, 1.0, .cpu)
	storage_set<f64>(s, 4, 2.0)
	offset := storage_offset<f64>(s, 3)
	match offset {
		CpuStorage<f64> {
			array := offset.to_array()
			assert array == [1.0, 2]
		}
		else {
			panic('This should not happen')
		}
	}
}
