module vtl

// DType describes the element types currently supported by VTL tensors.
pub enum DType {
	boolean
	int8
	int16
	int64
	native_int
	uint8
	uint16
	uint32
	uint64
	float32
	float64
	string
}

// dtype_of returns the VTL dtype corresponding to the compile-time type T.
pub fn dtype_of[T]() DType {
	$if T is bool {
		return .boolean
	} $else $if T is i8 {
		return .int8
	} $else $if T is i16 {
		return .int16
	} $else $if T is i64 {
		return .int64
	} $else $if T is int {
		return .native_int
	} $else $if T is u8 {
		return .uint8
	} $else $if T is u16 {
		return .uint16
	} $else $if T is u32 {
		return .uint32
	} $else $if T is u64 {
		return .uint64
	} $else $if T is f32 {
		return .float32
	} $else $if T is f64 {
		return .float64
	} $else $if T is string {
		return .string
	} $else {
		panic('${T.name} is not a supported VTL tensor dtype')
	}
}

// dtype returns the element dtype of the receiver tensor.
pub fn (t &Tensor[T]) dtype[T]() DType {
	return dtype_of[T]()
}

// promote_types returns the common dtype for two array dtypes. Its numeric
// rules are modeled on NumPy, using the integer and floating-point types VTL
// currently supports. If VTL has no signed integer width that can represent
// both inputs, it promotes to float64. Mixing strings with other dtypes is
// rejected; scalar-value weak promotion is not part of this dtype-only API.
pub fn promote_types(a DType, b DType) !DType {
	if a == b {
		return a
	}
	if a == .string || b == .string {
		return error('cannot promote string dtype with a different dtype')
	}
	if a == .boolean {
		return b
	}
	if b == .boolean {
		return a
	}
	if a == .float64 || b == .float64 {
		return .float64
	}
	if a == .float32 || b == .float32 {
		integer_dtype := if a == .float32 { b } else { a }
		if integer_dtype == .float32 || dtype_bits(integer_dtype) <= 24 {
			return .float32
		}
		return .float64
	}
	a_signed := is_signed_integer_dtype(a)
	b_signed := is_signed_integer_dtype(b)
	a_unsigned := is_unsigned_integer_dtype(a)
	b_unsigned := is_unsigned_integer_dtype(b)
	if a_signed && b_signed {
		return widest_signed_dtype(a, b)
	}
	if a_unsigned && b_unsigned {
		return widest_unsigned_dtype(a, b)
	}
	if (a_signed && b_unsigned) || (a_unsigned && b_signed) {
		signed := if a_signed { a } else { b }
		unsigned := if a_unsigned { a } else { b }
		signed_bits := dtype_bits(signed)
		unsigned_bits := dtype_bits(unsigned)
		if signed_bits > unsigned_bits {
			return signed
		}
		minimum_signed_bits := if signed_bits > unsigned_bits + 1 {
			signed_bits
		} else {
			unsigned_bits + 1
		}
		if wider_signed := smallest_signed_dtype(minimum_signed_bits) {
			return wider_signed
		}
		return .float64
	}
	return error('unsupported dtype promotion: ${a} and ${b}')
}

fn is_signed_integer_dtype(dtype DType) bool {
	return dtype == .int8 || dtype == .int16 || dtype == .int64 || dtype == .native_int
}

fn is_unsigned_integer_dtype(dtype DType) bool {
	return dtype == .uint8 || dtype == .uint16 || dtype == .uint32 || dtype == .uint64
}

fn dtype_bits(dtype DType) int {
	return match dtype {
		.int8, .uint8 { 8 }
		.int16, .uint16 { 16 }
		.uint32 { 32 }
		.int64, .uint64 { 64 }
		.native_int { int(sizeof(int) * 8) }
		else { 0 }
	}
}

fn widest_signed_dtype(a DType, b DType) DType {
	a_bits := dtype_bits(a)
	b_bits := dtype_bits(b)
	if a_bits > b_bits {
		return a
	}
	if b_bits > a_bits {
		return b
	}
	if a == .native_int || b == .native_int {
		return .native_int
	}
	return a
}

fn widest_unsigned_dtype(a DType, b DType) DType {
	if dtype_bits(a) >= dtype_bits(b) {
		return a
	}
	return b
}

fn smallest_signed_dtype(minimum_bits int) ?DType {
	candidates := [DType.int8, DType.int16, DType.native_int, DType.int64]
	mut best_bits := 65
	mut best := DType.native_int
	for candidate in candidates {
		bits := dtype_bits(candidate)
		if bits >= minimum_bits && bits < best_bits {
			best_bits = bits
			best = candidate
		}
	}
	if best_bits == 65 {
		return none
	}
	return best
}
