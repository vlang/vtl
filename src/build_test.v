module vtl

fn test_new_tensor() {
	t := new_tensor(1.0, [3])
	assert t.size() == 3
}
