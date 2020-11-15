module vtl

fn test_cpu_storage_with_default() {
	s := new_storage<f64>({
		len: 2
		init: voidptr(0)
		strategy: .cpu
	})
}
