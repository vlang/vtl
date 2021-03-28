module vtl

pub const (
	default_type = 'f64'
	default_init = Num(f64(0.0))
	default_size = int(sizeof(f64))
)

// `Num` is a sum type that lists the possible types to be used to define tensor values
pub type Num = byte | f32 | f64 | i16 | i64 | i8 | int | u16 | u32 | u64

// as_type<T> returns a Num casted to a given T type
pub fn (f Num) as_type<T>() T {
	return num_as_type<T>(f)
}

// formats a floating point value to be "pretty"
fn (val Num) prettify(notation bool) string {
	if notation {
		return val.strsci(3)
	} else {
		unsafe {
			return val.str()
		}
	}
}

// return a string of the input if f64 or f32 in scientific notation
// with digit_num deciamals displayed, max 17 digits
pub fn (f Num) strsci(digit_num int) string {
	match f {
		byte { return f.str() }
		u16 { return f.str() }
		u32 { return f.str() }
		u64 { return f.str() }
		i8 { return f.str() }
		i16 { return f.str() }
		int { return f.str() }
		i64 { return f.str() }
		f32 { return f.strsci(digit_num) }
		f64 { return f.strsci(digit_num) }
	}
}

// num_as_type<T> returns a Num casted to a given T type
pub fn num_as_type<T>(f Num) T {
	match f {
		byte {
			val := byte(f)
			return T(val)
		}
		u16 {
			val := u16(f)
			return T(val)
		}
		u32 {
			val := u32(f)
			return T(val)
		}
		u64 {
			val := u64(f)
			return T(val)
		}
		i8 {
			val := i8(f)
			return T(val)
		}
		i16 {
			val := i16(f)
			return T(val)
		}
		int {
			val := int(f)
			return T(val)
		}
		i64 {
			val := i64(f)
			return T(val)
		}
		f32 {
			val := f32(f)
			return T(val)
		}
		f64 {
			val := f64(f)
			return T(val)
		}
	}
}

// etype returns the string representation of the specific type of f
pub fn (f Num) etype() string {
	return typeof(f)
}

// esize returns the int representation of the specific size of f
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
	}
}

// esize returns a safe pointer to a given Num
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
	}
}

// str  returns the string representation for a given Num f
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
			return if str_f32.ends_with('.') { str_f32 + '0' } else { str_f32 }
		}
		f64 {
			str_f64 := f.str()
			return if str_f64.ends_with('.') { str_f64 + '0' } else { str_f64 }
		}
	}
}

// ptr_to_val_of_type returns the number obtained from ptr
pub fn ptr_to_val_of_type(ptr voidptr, t string) Num {
	match t {
		'byte' {
			val := unsafe { *(&byte(ptr)) }
			return Num(val)
		}
		'u16' {
			val := unsafe { *(&u16(ptr)) }
			return Num(val)
		}
		'u32' {
			val := unsafe { *(&u32(ptr)) }
			return Num(val)
		}
		'u64' {
			val := unsafe { *(&u64(ptr)) }
			return Num(val)
		}
		'i8' {
			val := unsafe { *(&i8(ptr)) }
			return Num(val)
		}
		'i16' {
			val := unsafe { *(&i16(ptr)) }
			return Num(val)
		}
		'int' {
			val := unsafe { *(&int(ptr)) }
			return Num(val)
		}
		'i64' {
			val := unsafe { *(&i64(ptr)) }
			return Num(val)
		}
		'f32' {
			val := unsafe { *(&f32(ptr)) }
			return Num(val)
		}
		'f64' {
			val := unsafe { *(&f64(ptr)) }
			return Num(val)
		}
		else {
			return Num(0.0)
		}
	}
}

// str_esize returns the int representation of a given type
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
		else { return 0 }
	}
}

// str_esize returns the int representation of the size for a given array of Num
pub fn arr_esize(arr []Num) int {
	if arr.len > 0 {
		return arr[0].esize()
	}
	return vtl.default_size
}

// str_etype returns the type for a given array of Num
pub fn arr_etype(arr []Num) string {
	if arr.len > 0 {
		return arr[0].etype()
	}
	return vtl.default_type
}
