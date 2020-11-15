module vtl

import vtl.storage

pub enum StorageStrategy {
	cpu
}

pub struct StorageData {
pub:
	len      int
	cap      int
	init     voidptr
	strategy StorageStrategy = .cpu
}

pub fn new_storage<T>(data StorageData) storage.CpuStorage {
	element_size := int(sizeof(T))
	if data.strategy == .cpu {
		return storage.new_cpu_with_default(data.len, data.cap, element_size, data.init)
	}
	return storage.new_cpu_with_default(data.len, data.cap, element_size, data.init)
}
