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
pub fn (a TensorDataType) string() string {
	match a {
		string { return a as string }
		else { return a.str() }
	}
}

// int uses `TensorDataType` as an integer.
pub fn (f TensorDataType) int() int {
	match f {
		int { return f }
		i64, f32, f64, bool { return int(f) }
		else { return 0 }
	}
}

// i64 uses `TensorDataType` as a 64-bit integer.
pub fn (f TensorDataType) i64() i64 {
	match f {
		i64 { return f }
		int, f32, f64, bool { return i64(f) }
		else { return 0 }
	}
}

// u8 uses `TensorDataType` as a 8-bit unsigned integer.
pub fn (f TensorDataType) u8() u8 {
	match f {
		u8 { return f }
		else { return 0 }
	}
}

// u16 uses `TensorDataType` as a 16-bit unsigned integer.
pub fn (f TensorDataType) u16() u16 {
	match f {
		u16 { return f }
		u8 { return u16(f) }
		else { return 0 }
	}
}

// u32 uses `TensorDataType` as a 32-bit unsigned integer.
pub fn (f TensorDataType) u32() u32 {
	match f {
		u32 { return f }
		int, f32, bool { return u32(f) }
		else { return 0 }
	}
}

// u64 uses `TensorDataType` as a 64-bit unsigned integer.
pub fn (f TensorDataType) u64() u64 {
	match f {
		u64 { return f }
		int, i64, f32, f64, bool { return u64(f) }
		else { return 0 }
	}
}

// f32 uses `TensorDataType` as a 32-bit float.
pub fn (f TensorDataType) f32() f32 {
	match f {
		f32 { return f }
		int, i64, f64 { return f32(f) }
		else { return 0.0 }
	}
}

// f64 uses `TensorDataType` as a float.
pub fn (f TensorDataType) f64() f64 {
	match f {
		f64 { return f }
		int, i64, f32 { return f64(f) }
		else { return 0.0 }
	}
}

// bool uses `TensorDataType` as a bool
pub fn (f TensorDataType) bool() bool {
	match f {
		bool { return f }
		string { return f.bool() }
		else { return false }
	}
}
