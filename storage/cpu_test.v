module storage

fn test_cpu_with_default() {
	init := 1.0
	c := new_cpu_with_default(2, 0, int(sizeof(f64)), &init)
	varray := cpu_to_varray<f64>(c)
	assert varray.len == 2
	assert varray[0] == 1.0
}
