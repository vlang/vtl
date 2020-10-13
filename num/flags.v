module num

import strings

struct ArrayFlags {
pub mut:
	contiguous bool
	fortran    bool
	owndata    bool
	write      bool
}

// default_flags returns a generic set of flags given a memory layout
// for an array.  This does not take into account a 1D case, so it
// is still safer to update flags afterwards.
pub fn default_flags(order string, ndims int) ArrayFlags {
	mut m := ArrayFlags{
		contiguous: false
		fortran: false
		owndata: true
		write: true
	}
	if order == 'F' {
		m.fortran = true
		if ndims == 1 {
			m.contiguous = true
		}
	} else if order == 'C' {
		m.contiguous = true
		if ndims == 1 {
			m.fortran = true
		}
	}
	return m
}

// str() returns the string representation of an ArrayFlags struct,
// providing helpful information about the memory layout of an ndarray
pub fn (f ArrayFlags) str() string {
	mut io := strings.new_builder(1000)
	io.write('C_CONTIGUOUS: ')
	io.write(f.contiguous.str())
	io.write('\nF_CONTIGUOUS: ')
	io.write(f.fortran.str())
	io.write('\nOWNDATA: ')
	io.write(f.owndata.str())
	io.write('\nWRITE: ')
	io.write(f.write.str())
	return io.str()
}

// all_flags returns an ArrayFlags object with all the flags set to true,
// helpful for updating the flags of an existing array, to compare
// against the existing flagmask
pub fn all_flags() ArrayFlags {
	m := ArrayFlags{
		contiguous: true
		fortran: true
		owndata: true
		write: true
	}
	return m
}

// no_flags returns an ArrayFlags object with no flags set to true,
// helpful for broadcasting methods where the result is read only
// and non-contiguous
pub fn no_flags() ArrayFlags {
	m := ArrayFlags{
		contiguous: false
		fortran: false
		owndata: false
		write: false
	}
	return m
}

// dup_flags returns a duplicated set of flags from an existing
// ArrayFlags object
pub fn dup_flags(f ArrayFlags) ArrayFlags {
	ret := ArrayFlags{
		contiguous: f.contiguous
		fortran: f.fortran
		owndata: f.owndata
		write: f.write
	}
	return ret
}

// update_flags updates the flags of an ndarray, taking into account
// a provided flagmask.  Checks for the memory layout of the underlying
// data and updates flags accordingly
pub fn (mut n NdArray) update_flags(mask ArrayFlags) {
	if mask.fortran && n.flags.fortran {
		if is_fortran_contiguous(n.shape, n.strides, n.ndims) {
			n.flags.fortran = true
			if n.ndims > 1 {
				n.flags.contiguous = false
			}
		} else {
			n.flags.fortran = false
		}
	}
	if mask.contiguous && n.flags.contiguous {
		if is_contiguous(n.shape, n.strides, n.ndims) {
			n.flags.contiguous = true
			if n.ndims > 1 {
				n.flags.fortran = false
			}
		} else {
			n.flags.contiguous = false
		}
	}
}
