module dtypes

import strconv

// Family is the NumPy dtype family of a descriptor.
pub enum Family {
	boolean
	integer
	unsigned
	float
	complex
	bytes
	unicode
	datetime
	timedelta
	object
	void
	unknown
}

// Descriptor is a parsed NumPy dtype descriptor such as `<i4`, `|S5`, or
// `<M8[ns]`. `width` is measured in bytes for numeric/bytes/void dtypes and in
// code points for unicode, matching NumPy's descriptor syntax.
pub struct Descriptor {
pub:
	raw        string
	byte_order u8
	kind       u8
	width      int
	unit       string
	family     Family
}

// parse decodes a NumPy dtype descriptor string.
pub fn parse(text string) !Descriptor {
	if text.len < 2 {
		return error('dtype: empty descriptor')
	}
	order := text[0]
	if order !in [`<`, `>`, `=`, `|`] {
		return error('dtype: invalid byte order in ${text}')
	}
	kind := text[1]
	rest := text[2..]
	mut width := 0
	mut unit := ''
	if kind in [`M`, `m`] {
		open := rest.index('[') or { return error('dtype: missing datetime unit in ${text}') }
		close := rest.index(']') or { return error('dtype: unterminated datetime unit in ${text}') }
		if close <= open + 1 {
			return error('dtype: empty datetime unit in ${text}')
		}
		unit = rest[open + 1..close]
		width = strconv.atoi(rest[..open]) or { return error('dtype: invalid width in ${text}') }
		if width != 8 {
			return error('dtype: datetime width must be 8, got ${width}')
		}
	} else if rest.len > 0 {
		width = strconv.atoi(rest) or { return error('dtype: invalid width in ${text}') }
		if width < 0 {
			return error('dtype: negative width in ${text}')
		}
	}
	family := family_for(kind)
	if family == .unknown {
		return error('dtype: unknown kind `${rune(kind)}` in ${text}')
	}
	return Descriptor{
		raw:        text
		byte_order: order
		kind:       kind
		width:      width
		unit:       unit
		family:     family
	}
}

fn family_for(kind u8) Family {
	return match kind {
		`b` { .boolean }
		`i` { .integer }
		`u` { .unsigned }
		`f` { .float }
		`c` { .complex }
		`S` { .bytes }
		`U` { .unicode }
		`M` { .datetime }
		`m` { .timedelta }
		`O` { .object }
		`V` { .void }
		else { .unknown }
	}
}

// is_numeric reports whether the descriptor belongs to a numeric family.
pub fn (d &Descriptor) is_numeric() bool {
	return match d.family {
		.boolean, .integer, .unsigned, .float, .complex { true }
		else { false }
	}
}

// label returns a human-readable family name for diagnostics.
pub fn (d &Descriptor) label() string {
	return match d.family {
		.boolean { 'boolean' }
		.integer { 'signed integer' }
		.unsigned { 'unsigned integer' }
		.float { 'floating point' }
		.complex { 'complex' }
		.bytes { 'fixed-width bytes' }
		.unicode { 'fixed-width unicode string' }
		.datetime { 'datetime' }
		.timedelta { 'timedelta' }
		.object { 'object' }
		.void { 'structured/void' }
		.unknown { 'unknown' }
	}
}

// ensure_supported enforces the VTL boundary: tensors are homogeneous and
// numeric. Non-numeric families are rejected with an explicit reason rather
// than being coerced into numbers.
pub fn (d &Descriptor) ensure_supported() ! {
	if !d.is_numeric() {
		detail := if d.unit.len > 0 { ' (unit ${d.unit})' } else { '' }
		return error('dtype ${d.raw} (${d.label()}${detail}) is not supported by VTL tensors; use a companion table/array package')
	}
	valid_width := match d.family {
		.boolean { d.width == 1 }
		.integer, .unsigned { d.width in [1, 2, 4, 8] }
		.float { d.width in [4, 8] }
		.complex { d.width == 16 }
		else { false }
	}
	if !valid_width {
		return error('dtype ${d.raw} has unsupported width ${d.width} for ${d.label()}')
	}
}

// supported_families lists the families VTL tensors accept.
pub fn supported_families() []Family {
	return [.boolean, .integer, .unsigned, .float, .complex]
}

// excluded_families lists families that are intentionally out of scope.
pub fn excluded_families() []Family {
	return [.bytes, .unicode, .datetime, .timedelta, .object, .void]
}
