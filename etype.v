module vtl

// `Num` is a sum type that lists the possible types to be decoded and used
pub type Num = any_float | any_int | byte | f32 | f64 | i16 | i64 | i8 | int | u16 | u32 |
	u64

pub fn (f Num) as_type<T>() T {
	return f as T
}

pub fn (f Num) etype() string {
	return typeof(f)
}

pub fn (f Num) esize() string {
	match f {
		byte { return sizeof(f).str() }
		u16 { return sizeof(f).str() }
		u32 { return sizeof(f).str() }
		u64 { return sizeof(f).str() }
		i8 { return sizeof(f).str() }
		i16 { return sizeof(f).str() }
		int { return sizeof(f).str() }
		i64 { return sizeof(f).str() }
		f32 { return sizeof(f).str() }
		f64 { return sizeof(f).str() }
		any_int { return sizeof(f).str() }
		any_float { return sizeof(f).str() }
	}
}

pub fn (f Num) ptr() voidptr {
	match f {
		byte {
			val := byte(0)
			return voidptr(&val)
		}
		u16 {
			val := u16(0)
			return voidptr(&val)
		}
		u32 {
			val := u32(0)
			return voidptr(&val)
		}
		u64 {
			val := u64(0)
			return voidptr(&val)
		}
		i8 {
			val := i8(0)
			return voidptr(&val)
		}
		i16 {
			val := i16(0)
			return voidptr(&val)
		}
		int {
			val := int(0)
			return voidptr(&val)
		}
		i64 {
			val := i64(0)
			return voidptr(&val)
		}
		f32 {
			val := f32(0.)
			return voidptr(&val)
		}
		f64 {
			val := f64(0.)
			return voidptr(&val)
		}
		any_int {
			val := any_int(0)
			return voidptr(&val)
		}
		any_float {
			val := any_float(0.)
			return voidptr(&val)
		}
	}
}

pub fn ptr_to_val_of_type(ptr voidptr, t string) Num {
	match t {
		'byte' {
			val := unsafe {*(&byte(&ptr[0]))}
			return Num(val)
		}
		'u16' {
			val := unsafe {*(&u16(&ptr[0]))}
			return Num(val)
		}
		'u32' {
			val := unsafe {*(&u32(&ptr[0]))}
			return Num(val)
		}
		'u64' {
			val := unsafe {*(&u64(&ptr[0]))}
			return Num(val)
		}
		'i8' {
			val := unsafe {*(&i8(&ptr[0]))}
			return Num(val)
		}
		'i16' {
			val := unsafe {*(&i16(&ptr[0]))}
			return Num(val)
		}
		'int' {
			val := unsafe {*(&int(&ptr[0]))}
			return Num(val)
		}
		'i64' {
			val := unsafe {*(&i64(&ptr[0]))}
			return Num(val)
		}
		'f32' {
			val := unsafe {*(&f32(&ptr[0]))}
			return Num(val)
		}
		'f64' {
			val := unsafe {*(&f64(&ptr[0]))}
			return Num(val)
		}
		'any_int' {
			val := unsafe {*(&any_int(&ptr[0]))}
			return Num(val)
		}
		'any_float' {
			val := unsafe {*(&any_float(&ptr[0]))}
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
	match typeof(arr) {
		'array_byte' { return int(sizeof(byte)) }
		'array_u16' { return int(sizeof(u16)) }
		'array_u32' { return int(sizeof(u32)) }
		'array_u64' { return int(sizeof(u64)) }
		'array_i8' { return int(sizeof(i8)) }
		'array_i16' { return int(sizeof(i16)) }
		'array_int' { return int(sizeof(int)) }
		'array_i64' { return int(sizeof(i64)) }
		'array_f32' { return int(sizeof(f32)) }
		'array_f64' { return int(sizeof(f64)) }
		'array_any_int' { return int(sizeof(any_int)) }
		'array_any_float' { return int(sizeof(any_float)) }
		else { return 0 }
	}
}

pub fn arr_etype(arr []Num) string {
	match typeof(arr) {
		'array_byte' { return 'byte' }
		'array_u16' { return 'u16' }
		'array_u32' { return 'u32' }
		'array_u64' { return 'u64' }
		'array_i8' { return 'i8' }
		'array_i16' { return 'i16' }
		'array_int' { return 'int' }
		'array_i64' { return 'i64' }
		'array_f32' { return 'f32' }
		'array_f64' { return 'f64' }
		'array_any_int' { return 'any_int' }
		'array_any_float' { return 'any_float' }
		else { return '' }
	}
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
