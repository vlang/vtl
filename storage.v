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

[inline]
pub fn new_storage<T>(data StorageData) storage.CpuStorage {
	element_size := int(sizeof(T))
	return create_storage(data.len, data.cap, element_size, data.init, data.strategy)
}

[inline]
pub fn new_storage_like(s storage.CpuStorage) storage.CpuStorage {
	return create_storage(s.len, s.capacity, s.element_size, voidptr(0), storage_strategy(s))
}

[inline]
pub fn new_storage_from_varray<T>(arr []T, strategy StorageStrategy) storage.CpuStorage {
	return create_storage_from_c_array(arr.len, 0, int(sizeof(T)), arr.data, strategy)
}

pub fn storage_to_varray<T>(s storage.CpuStorage) []T {
	if s.element_size == int(sizeof(T)) {
		mut arr := []T{}
		arr.push_many(s.data, s.len)
		return arr
	}
	panic('CpuStorage.to_varray<T>: incoming type T does not match with the stored data type')
}

fn create_storage(len int, cap int, element_size int, init voidptr, strategy StorageStrategy) storage.CpuStorage {
	if strategy == .cpu {
		return storage.new_cpu_with_default(len, cap, element_size, init)
	}
	return storage.new_cpu_with_default(len, cap, element_size, init)
}

fn create_storage_from_c_array(len int, cap int, element_size int, c_array voidptr, strategy StorageStrategy) storage.CpuStorage {
	if strategy == .cpu {
		return storage.new_cpu_from_c_array(len, cap, element_size, c_array)
	}
	return storage.new_cpu_from_c_array(len, cap, element_size, c_array)
}

fn storage_strategy(storage storage.CpuStorage) StorageStrategy {
	// @todo: change when more strategies are available
	return .cpu
}
