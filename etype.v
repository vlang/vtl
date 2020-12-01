module vtl

pub const (
	default_type = 'f64'
	default_init = Num(f64(0.0))
	default_size = int(sizeof(f64))
)

// `Num` is a sum type that lists the possible types to be decoded and used
pub type Num = any_float | any_int | byte | f32 | f64 | i16 | i64 | i8 | int | u16 | u32 |
	u64

pub fn (f Num) as_type<T>() T {
	return f as T
}

pub fn (f Num) etype() string {
	return typeof(f)
}

pub fn (f Num) esize() int {
	match f {
		byte { return int(sizeof(f)) }
		u16 { return int(sizeof(f)) }
		u32 { return int(sizeof(f)) }
		u64 { return int(sizeof(f)) }
		i8 { return int(sizeof(f)) }
		i16 { return int(sizeof(f)) }
		int { return int(sizeof(f)) }
		i64 { return int(sizeof(f)) }
		f32 { return int(sizeof(f)) }
		f64 { return int(sizeof(f)) }
		any_int { return int(sizeof(f)) }
		any_float { return int(sizeof(f)) }
	}
}

pub fn (f Num) ptr() voidptr {
	match f {
		byte { return voidptr(&f) }
		u16 { return voidptr(&f) }
		u32 { return voidptr(&f) }
		u64 { return voidptr(&f) }
		i8 { return voidptr(&f) }
		i16 { return voidptr(&f) }
		int { return voidptr(&f) }
		i64 { return voidptr(&f) }
		f32 { return voidptr(&f) }
		f64 { return voidptr(&f) }
		any_int { return voidptr(&f) }
		any_float { return voidptr(&f) }
	}
}

pub fn ptr_to_val_of_type(ptr voidptr, t string) Num {
	match t {
		'byte' {
			val := unsafe {*(&byte(ptr))}
			return Num(val)
		}
		'u16' {
			val := unsafe {*(&u16(ptr))}
			return Num(val)
		}
		'u32' {
			val := unsafe {*(&u32(ptr))}
			return Num(val)
		}
		'u64' {
			val := unsafe {*(&u64(ptr))}
			return Num(val)
		}
		'i8' {
			val := unsafe {*(&i8(ptr))}
			return Num(val)
		}
		'i16' {
			val := unsafe {*(&i16(ptr))}
			return Num(val)
		}
		'int' {
			val := unsafe {*(&int(ptr))}
			return Num(val)
		}
		'i64' {
			val := unsafe {*(&i64(ptr))}
			return Num(val)
		}
		'f32' {
			val := unsafe {*(&f32(ptr))}
			return Num(val)
		}
		'f64' {
			val := unsafe {*(&f64(ptr))}
			return Num(val)
		}
		'any_int' {
			val := unsafe {*(&any_int(ptr))}
			return Num(val)
		}
		'any_float' {
			val := unsafe {*(&any_float(ptr))}
			return Num(val)
		}
		else {
			return Num(0.0)
		}
	}
}

pub fn str_esize(t string) int {
	match t {
		'byte' { return int(sizeof(byte)) }
		'u16' { return int(sizeof(u16)) }
		'u32' { return int(sizeof(u32)) }
		'u64' { return int(sizeof(u64)) }
		'i8' { return int(sizeof(i8)) }
		'i16' { return int(sizeof(i16)) }
		'int' { return int(sizeof(int)) }
		'i64' { return int(sizeof(i64)) }
		'f32' { return int(sizeof(f32)) }
		'f64' { return int(sizeof(f64)) }
		'any_int' { return int(sizeof(any_int)) }
		'any_float' { return int(sizeof(any_float)) }
		else { return 0 }
	}
}

pub fn arr_esize(arr []Num) int {
	if arr.len > 0 {
		return arr[0].esize()
	}
	return default_size
}

pub fn arr_etype(arr []Num) string {
	if arr.len > 0 {
		return arr[0].etype()
	}
	return default_type
}

pub fn (f Num) str() string {
	match f {
		byte {
			return f.str()
		}
		u16 {
			return f.str()
		}
		u32 {
			return f.str()
		}
		u64 {
			return f.str()
		}
		i8 {
			return f.str()
		}
		i16 {
			return f.str()
		}
		int {
			return f.str()
		}
		i64 {
			return f.str()
		}
		f32 {
			str_f32 := f.str()
			return if str_f32.ends_with('.') {
				str_f32 + '0'
			} else {
				str_f32
			}
		}
		f64 {
			str_f64 := f.str()
			return if str_f64.ends_with('.') {
				str_f64 + '0'
			} else {
				str_f64
			}
		}
		any_int {
			return f.str()
		}
		any_float {
			return f.str()
		}
	}
}
