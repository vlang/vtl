module vtl

fn test_tensor() {
	t := tensor(1.0, [3])
	assert t.size() == 3
}
