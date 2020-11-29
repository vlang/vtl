module vtl

fn test_cpu_storage_with_default() {
	init := 1.0
	s := new_storage<f64>({
		len: 2
		init: &init
		strategy: .cpu
	})
	varray := storage_to_varray<f64>(s)
	assert varray.len == 2
	assert varray[0] == 1.0
}

fn test_cpu_storage_from_varray() {
	s := new_storage_from_varray([1.0, 2.0], .cpu)
	varray := storage_to_varray<f64>(s)
	assert varray.len == 2
	assert varray[1] == 2.0
}
