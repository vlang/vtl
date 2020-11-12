import vtl.num

fn test_dup_flags() {
	mut a := num.all_flags()
	a.fortran = false
	b := num.dup_flags(a)
	assert a.owndata == b.owndata
	assert a.contiguous == b.contiguous
	assert a.fortran == b.fortran
	assert a.write == b.write
}

fn test_update_flags_fortran() {
	mut a := num.allocate_cpu([3, 3], 'F')
	a.update_flags(num.all_flags())
	assert a.flags.fortran == true
	assert a.flags.contiguous == false
}

fn test_update_flags_contiguous() {
	mut a := num.allocate_cpu([3, 3], 'C')
	a.update_flags(num.all_flags())
	assert a.flags.fortran == false
	assert a.flags.contiguous == true
}
