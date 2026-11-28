module dtypes

import vtl.dtypes as dt

fn test_parse_numeric_descriptors() ! {
	i4 := dt.parse('<i4')!
	assert i4.family == .integer
	assert i4.width == 4
	assert i4.byte_order == `<`
	assert i4.is_numeric()

	assert dt.parse('|b1')!.family == .boolean
	assert dt.parse('>u2')!.family == .unsigned
	assert dt.parse('<f8')!.family == .float
	assert dt.parse('<c16')!.family == .complex
}

fn test_parse_non_numeric_descriptors() ! {
	bytes := dt.parse('|S5')!
	assert bytes.family == .bytes
	assert bytes.width == 5
	assert !bytes.is_numeric()
	assert bytes.label() == 'fixed-width bytes'

	unicode := dt.parse('<U3')!
	assert unicode.family == .unicode
	assert unicode.width == 3

	assert dt.parse('<M8[ns]')!.family == .datetime
	assert dt.parse('<m8[us]')!.unit == 'us'
	assert dt.parse('|O')!.family == .object
	assert dt.parse('|V4')!.family == .void
}

fn test_datetime_keeps_unit_and_requires_width_eight() ! {
	dt_desc := dt.parse('<M8[ms]')!
	assert dt_desc.unit == 'ms'
	assert dt_desc.width == 8

	dt.parse('<M4[ns]') or {
		assert err.msg().contains('width must be 8')
		return
	}
	assert false, 'datetime width other than 8 should be rejected'
}

fn test_ensure_supported_accepts_numeric_families() ! {
	for descriptor in ['|b1', '<i4', '>u8', '<f8', '<c16'] {
		dt.parse(descriptor)!.ensure_supported()!
	}
}

fn test_ensure_supported_rejects_non_numeric_families() ! {
	cases := [
		['|S5', 'fixed-width bytes']
		['<U3', 'fixed-width unicode string']
		['<M8[ns]', 'datetime']
		['<m8[us]', 'timedelta']
		['|O', 'object']
		['|V4', 'structured/void'],
	]
	for case in cases {
		descriptor := dt.parse(case[0])!
		descriptor.ensure_supported() or {
			assert err.msg().contains(case[1])
			assert err.msg().contains('companion table/array package')
			continue
		}
		assert false, '${case[0]} should be rejected'
	}
}

fn test_ensure_supported_rejects_unsupported_numeric_widths() ! {
	dt.parse('<f2')!.ensure_supported() or {
		assert err.msg().contains('unsupported width')
		return
	}
	assert false, 'f2 has no VTL tensor type'
}

fn test_parse_rejects_malformed_descriptors() ! {
	dt.parse('') or {
		assert err.msg().contains('empty descriptor')
		return
	}
	assert false, 'empty descriptor should be rejected'
}

fn test_parse_rejects_unknown_kind() ! {
	dt.parse('|X4') or {
		assert err.msg().contains('unknown kind')
		return
	}
	assert false, 'unknown kind should be rejected'
}

fn test_parse_rejects_missing_datetime_unit() ! {
	dt.parse('<M8') or {
		assert err.msg().contains('missing datetime unit')
		return
	}
	assert false, 'datetime without unit should be rejected'
}

fn test_family_lists_are_disjoint() ! {
	supported := dt.supported_families()
	excluded := dt.excluded_families()
	assert supported.len == 5
	assert excluded.len == 6
	for family in supported {
		assert family !in excluded
	}
}
