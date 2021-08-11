module storage

// StorageData is a sum type that lists the possible types to be used to define storage
pub type StorageDataType = byte | f32 | f64 | i16 | i64 | i8 | int | u16 | u32 | u64

pub interface Storage {}

pub enum StorageStrategy {
	cpu
}

pub fn new_storage<T>(len int, cap int, init T, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return new_cpu<T>(len, cap, init)
	}
	return new_cpu<T>(len, cap, init)
}

pub fn from_array<T>(arr []T, strategy StorageStrategy) Storage {
	if strategy == .cpu {
		return cpu_from_array<T>(arr)
	}
	return cpu_from_array<T>(arr)
}

pub fn storage_clone<T>(s Storage) Storage {
	match s {
		CpuStorage<T> {
			return s.clone()
		}
		else {
			panic('unsupported storage type')
		}
	}
}

pub fn storage_like<T>(s Storage) Storage {
	match s {
		CpuStorage<T> {
			return s.like()
		}
		else {
			panic('unsupported storage type')
		}
	}
}

pub fn storage_get<T>(s Storage, index int) T {
	match s {
		CpuStorage<T> {
			return s.get(index)
		}
		else {
			panic('unsupported storage type')
		}
	}
}

pub fn storage_fill<T>(s Storage, val T) {
	match mut s {
		CpuStorage<T> {
			for i in 0 .. s.len {
				s.data[i] = val
			}
		}
		else {
			panic('unsupported storage type')
		}
	}
}

pub fn storage_set<T>(s Storage, index int, val T) {
	match mut s {
		CpuStorage<T> {
			s.data[index] = val
		}
		else {
			panic('unsupported storage type')
		}
	}
}
