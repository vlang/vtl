module vtl

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t &Tensor<T>) get<T>(index []int) T {
	offset := t.offset_index(index)
	return t.data.get<T>(offset)
}

// offset_index returns the index to a Tensor's data at
// a given index
[inline]
pub fn (t &Tensor<T>) offset_index<T>(index []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
	}
	return offset
}

// nth_index returns the nth index of a Tensor's shape
[inline]
pub fn (t &Tensor<T>) nth_index<T>(n int) []int {
        rank := t.rank()
        mut index := []int{len: rank}
        for i in 0 .. rank {
                index[i] = 0
        }
        mut i := 0
        for {
                if i == n {
                        return index
                }
                i += 1
                index[0] += 1
                for j := 0; j < rank; j++ {
                        if index[j] == t.shape[j] {
                                index[j] = 0
                                if j < rank - 1 {
                                        index[j + 1] += 1
                                }
                        }
                }
        }

        return index
}

// strided_offset_index returns the index of the starting offset
// for arrays that may be negatively strided
pub fn (t &Tensor<T>) strided_offset_index<T>() int {
	mut offset := 0
	for i in 0 .. t.rank() {
		if t.strides[i] < 0 {
			offset += (t.shape[i] - 1) * int(fabs(t.strides[i]))
		}
	}
	return offset
}
