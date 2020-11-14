module storage

pub enum StorageStrategy {
	cpu
}

pub interface Storage {
	get(i int) voidptr
	set(i int, val voidptr)
}

pub struct StorageData {
	cap      int
	init     voidptr
	strategy StorageStrategy
}

pub fn new_storage<T>(data StorageData) CpuStorage {
	cap := data.cap
	match data.strategy {
		.cpu { return new_cpu_with_default<T>(cap) }
		else { return new_cpu_with_default<T>(cap) }
	}
}
