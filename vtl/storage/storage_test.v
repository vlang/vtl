module storage

fn test_cpu_storage_with_default() {
	s := new_storage(len: 2, init: 1.0, strategy: .cpu)
	varray := storage_to_varray<f64>(s)
	assert varray.len == 2
	assert varray[0] == 1.0
}
