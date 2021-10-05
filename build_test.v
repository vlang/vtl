module vtl

fn test_new_tensor() {
	t := new_tensor<f64>(1.0, [3], .row_major)
	assert t.size() == 3
}
