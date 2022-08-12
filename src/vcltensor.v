module vtl

import vtl.storage

// VclTensor is the main structure defined by VTL to manage N Dimensional data
[heap]
pub struct VclTensor<T> {
pub mut:
	data    &storage.VclStorage<T>
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// vcl
pub fn (t &Tensor<T>) vcl(params storage.VclStorageParams) ?&VclTensor<T> {
	row_tensor := t.copy(.row_major)
	cldata := row_tensor.data.vcl(params)?
	return &VclTensor<T>{
		data: cldata
		memory: row_tensor.memory
		size: row_tensor.size
		shape: row_tensor.shape
		strides: row_tensor.strides
	}
}
