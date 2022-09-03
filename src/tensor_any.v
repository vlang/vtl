module vtl

// TensorDataType is a sum type that lists the possible types to be used to define storage
pub type TensorDataType = bool
	| byte
	| f32
	| f64
	| i16
	| i64
	| i8
	| int
	| string
	| u16
	| u32
	| u64
	| u8

pub fn new_td<T>(x T) TensorDataType {
	$if T is bool {
		return TensorDataType(x)
	} $else $if T is byte {
		return TensorDataType(x)
	} $else $if T is f32 {
		return TensorDataType(x)
	} $else $if T is f64 {
		return TensorDataType(x)
	} $else $if T is i16 {
		return TensorDataType(x)
	} $else $if T is i64 {
		return TensorDataType(x)
	} $else $if T is i8 {
		return TensorDataType(x)
	} $else $if T is int {
		return TensorDataType(x)
	} $else $if T is string {
		return TensorDataType(x)
	} $else $if T is u8 {
		return TensorDataType(x)
	} $else $if T is u16 {
		return TensorDataType(x)
	} $else $if T is u32 {
		return TensorDataType(x)
	} $else $if T is u64 {
		return TensorDataType(x)
	} $else {
		panic('${typeof(x).name} is not a supported type for a Tensor. Check the type TensorDataType to know the valid data types')
	}
}

pub fn new_t<T>(x TensorDataType) T {
	$if T is bool {
		return x.bool()
	} $else $if T is byte {
		return x.byte()
	} $else $if T is f32 {
		return x.f32()
	} $else $if T is f64 {
		return x.f64()
	} $else $if T is i16 {
		return x.i16()
	} $else $if T is i64 {
		return x.i64()
	} $else $if T is i8 {
		return x.i8()
	} $else $if T is int {
		return x.int()
	} $else $if T is string {
		return x.str()
	} $else $if T is u8 {
		return x.u8()
	} $else $if T is u16 {
		return x.u16()
	} $else $if T is u32 {
		return x.u32()
	} $else $if T is u64 {
		return x.u64()
	} $else {
		panic('$T.name is not a supported type for a Tensor. Check the type TensorDataType to know the valid data types')
	}
}

// string returns `TensorDataType` as a string.
pub fn (v TensorDataType) string() string {
	return v.str()
}

// int uses `TensorDataType` as an integer.
pub fn (v TensorDataType) int() int {
	match v {
		int { return v }
		i64, f32, f64, bool { return int(v) }
		else { return 0 }
	}
}

// i64 uses `TensorDataType` as a 64-bit integer.
pub fn (v TensorDataType) i64() i64 {
	match v {
		i64 { return v }
		int, f32, f64, bool { return i64(v) }
		else { return 0 }
	}
}

// u8 uses `TensorDataType` as a 8-bit unsigned integer.
pub fn (v TensorDataType) u8() u8 {
	match v {
		u8 { return v }
		else { return 0 }
	}
}

// u16 uses `TensorDataType` as a 16-bit unsigned integer.
pub fn (v TensorDataType) u16() u16 {
	match v {
		u16 { return v }
		u8 { return u16(v) }
		else { return 0 }
	}
}

// u32 uses `TensorDataType` as a 32-bit unsigned integer.
pub fn (v TensorDataType) u32() u32 {
	match v {
		u32 { return v }
		int, f32, bool { return u32(v) }
		else { return 0 }
	}
}

// u64 uses `TensorDataType` as a 64-bit unsigned integer.
pub fn (v TensorDataType) u64() u64 {
	match v {
		u64 { return v }
		int, i64, f32, f64, bool { return u64(v) }
		else { return 0 }
	}
}

// f32 uses `TensorDataType` as a 32-bit float.
pub fn (v TensorDataType) f32() f32 {
	match v {
		f32 { return v }
		int, i64, f64 { return f32(v) }
		else { return 0.0 }
	}
}

// f64 uses `TensorDataType` as a float.
pub fn (v TensorDataType) f64() f64 {
	match v {
		f64 { return v }
		int, i64, f32 { return f64(v) }
		else { return 0.0 }
	}
}

// bool uses `TensorDataType` as a bool
pub fn (v TensorDataType) bool() bool {
	match v {
		bool { return v }
		string { return v.bool() }
		else { return false }
	}
}
