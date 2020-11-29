module vtl

// `Any` is a sum type that lists the possible types to be decoded and used
pub type Any = byte | u16 | u32 | u64 | i8 | i16 | int | i64 | f32 | f64 | any_int | any_float

pub fn (f Any) as_type<T>() T {
        return f as T
}

pub fn (f Any) type() string {
        return typeof(f)
}

pub fn (f Any) str() string {
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
