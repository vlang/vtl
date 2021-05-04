module vtl

fn test_cpu_storage_from_varray() {
	s := new_storage_from_varray<f64>([1.0, 2.0], .cpu)
	varray := storage_to_varray<f64>(s)
	assert varray.len == 2
	assert varray[1] == 2.0
}
