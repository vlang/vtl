module vtl

import math.complex as vcomplex

// TensorDataType is a sum type that lists the possible types to be used to define storage
pub type TensorDataType = bool
	| f32
	| f64
	| i16
	| i32
	| i64
	| i8
	| int
	| string
	| u16
	| u32
	| u64
	| u8

// td exposes this operation as part of the public API.
pub fn td[T](x T) TensorDataType {
	$if T is bool {
		return TensorDataType(x)
	} $else $if T is f32 {
		return TensorDataType(x)
	} $else $if T is f64 {
		return TensorDataType(x)
	} $else $if T is i16 {
		return TensorDataType(x)
	} $else $if T is i32 {
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

// cast exposes this operation as part of the public API.
pub fn cast[T](x TensorDataType) T {
	$if T is vcomplex.Complex {
		return T(vcomplex.Complex{ re: x.f64(), im: 0 })
	} $else $if T is bool {
		return x.bool()
	} $else $if T is f32 {
		return x.f32()
	} $else $if T is f64 {
		return x.f64()
	} $else $if T is i16 {
		return x.i16()
	} $else $if T is i32 {
		return x.i32()
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
		panic('${T.name} is not a supported type for a Tensor. Check the type TensorDataType to know the valid data types')
	}
}

// string returns `TensorDataType` as a string.
pub fn (v TensorDataType) string() string {
	return v.str()
}

// int converts a numeric TensorDataType value to int.
pub fn (v TensorDataType) int() int {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return v as int }
		i8 { return int(v as i8) }
		i16 { return int(v as i16) }
		i32 { return int(v as i32) }
		i64 { return int(v as i64) }
		u8 { return int(v as u8) }
		u16 { return int(v as u16) }
		u32 { return int(v as u32) }
		u64 { return int(v as u64) }
		f32 { return int(v as f32) }
		f64 { return int(v as f64) }
		string { return 0 }
	}
}

// i64 converts a numeric TensorDataType value to i64.
pub fn (v TensorDataType) i64() i64 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return i64(v as int) }
		i8 { return i64(v as i8) }
		i16 { return i64(v as i16) }
		i32 { return i64(v as i32) }
		i64 { return v as i64 }
		u8 { return i64(v as u8) }
		u16 { return i64(v as u16) }
		u32 { return i64(v as u32) }
		u64 { return i64(v as u64) }
		f32 { return i64(v as f32) }
		f64 { return i64(v as f64) }
		string { return 0 }
	}
}

// i32 converts a numeric TensorDataType value to i32.
pub fn (v TensorDataType) i32() i32 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return i32(v as int) }
		i8 { return i32(v as i8) }
		i16 { return i32(v as i16) }
		i32 { return v as i32 }
		i64 { return i32(v as i64) }
		u8 { return i32(v as u8) }
		u16 { return i32(v as u16) }
		u32 { return i32(v as u32) }
		u64 { return i32(v as u64) }
		f32 { return i32(v as f32) }
		f64 { return i32(v as f64) }
		string { return 0 }
	}
}

// i8 converts a numeric TensorDataType value to i8.
pub fn (v TensorDataType) i8() i8 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return i8(v as int) }
		i8 { return v as i8 }
		i16 { return i8(v as i16) }
		i32 { return i8(v as i32) }
		i64 { return i8(v as i64) }
		u8 { return i8(v as u8) }
		u16 { return i8(v as u16) }
		u32 { return i8(v as u32) }
		u64 { return i8(v as u64) }
		f32 { return i8(v as f32) }
		f64 { return i8(v as f64) }
		string { return 0 }
	}
}

// i16 converts a numeric TensorDataType value to i16.
pub fn (v TensorDataType) i16() i16 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return i16(v as int) }
		i8 { return i16(v as i8) }
		i16 { return v as i16 }
		i32 { return i16(v as i32) }
		i64 { return i16(v as i64) }
		u8 { return i16(v as u8) }
		u16 { return i16(v as u16) }
		u32 { return i16(v as u32) }
		u64 { return i16(v as u64) }
		f32 { return i16(v as f32) }
		f64 { return i16(v as f64) }
		string { return 0 }
	}
}

// u8 converts a numeric TensorDataType value to u8.
pub fn (v TensorDataType) u8() u8 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return u8(v as int) }
		i8 { return u8(v as i8) }
		i16 { return u8(v as i16) }
		i32 { return u8(v as i32) }
		i64 { return u8(v as i64) }
		u8 { return v as u8 }
		u16 { return u8(v as u16) }
		u32 { return u8(v as u32) }
		u64 { return u8(v as u64) }
		f32 { return u8(v as f32) }
		f64 { return u8(v as f64) }
		string { return 0 }
	}
}

// u16 converts a numeric TensorDataType value to u16.
pub fn (v TensorDataType) u16() u16 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return u16(v as int) }
		i8 { return u16(v as i8) }
		i16 { return u16(v as i16) }
		i32 { return u16(v as i32) }
		i64 { return u16(v as i64) }
		u8 { return u16(v as u8) }
		u16 { return v as u16 }
		u32 { return u16(v as u32) }
		u64 { return u16(v as u64) }
		f32 { return u16(v as f32) }
		f64 { return u16(v as f64) }
		string { return 0 }
	}
}

// u32 converts a numeric TensorDataType value to u32.
pub fn (v TensorDataType) u32() u32 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return u32(v as int) }
		i8 { return u32(v as i8) }
		i16 { return u32(v as i16) }
		i32 { return u32(v as i32) }
		i64 { return u32(v as i64) }
		u8 { return u32(v as u8) }
		u16 { return u32(v as u16) }
		u32 { return v as u32 }
		u64 { return u32(v as u64) }
		f32 { return u32(v as f32) }
		f64 { return u32(v as f64) }
		string { return 0 }
	}
}

// u64 converts a numeric TensorDataType value to u64.
pub fn (v TensorDataType) u64() u64 {
	match v {
		bool { return if v { 1 } else { 0 } }
		int { return u64(v as int) }
		i8 { return u64(v as i8) }
		i16 { return u64(v as i16) }
		i32 { return u64(v as i32) }
		i64 { return u64(v as i64) }
		u8 { return u64(v as u8) }
		u16 { return u64(v as u16) }
		u32 { return u64(v as u32) }
		u64 { return v as u64 }
		f32 { return u64(v as f32) }
		f64 { return u64(v as f64) }
		string { return 0 }
	}
}

// f32 converts a numeric TensorDataType value to f32.
pub fn (v TensorDataType) f32() f32 {
	match v {
		bool { return if v { 1.0 } else { 0.0 } }
		int { return f32(v as int) }
		i8 { return f32(v as i8) }
		i16 { return f32(v as i16) }
		i32 { return f32(v as i32) }
		i64 { return f32(v as i64) }
		u8 { return f32(v as u8) }
		u16 { return f32(v as u16) }
		u32 { return f32(v as u32) }
		u64 { return f32(v as u64) }
		f32 { return v as f32 }
		f64 { return f32(v as f64) }
		string { return 0.0 }
	}
}

// f64 converts a numeric TensorDataType value to f64.
pub fn (v TensorDataType) f64() f64 {
	match v {
		bool { return if v { 1.0 } else { 0.0 } }
		int { return f64(v as int) }
		i8 { return f64(v as i8) }
		i16 { return f64(v as i16) }
		i32 { return f64(v as i32) }
		i64 { return f64(v as i64) }
		u8 { return f64(v as u8) }
		u16 { return f64(v as u16) }
		u32 { return f64(v as u32) }
		u64 { return f64(v as u64) }
		f32 { return f64(v as f32) }
		f64 { return v as f64 }
		string { return 0.0 }
	}
}

// bool uses `TensorDataType` as a bool
pub fn (v TensorDataType) bool() bool {
	match v {
		bool { return v }
		string { return v.bool() }
		f32 { return v != f32(0) }
		f64 { return v != f64(0) }
		i8 { return v != i8(0) }
		i16 { return v != i16(0) }
		i32 { return v != i32(0) }
		i64 { return v != i64(0) }
		int { return v != int(0) }
		u8 { return v != u8(0) }
		u16 { return v != u16(0) }
		u32 { return v != u32(0) }
		u64 { return v != u64(0) }
	}
}
