module num

// concatenates two ndarrays together, this has to be here
// and not in num since its needed for print.
pub fn concatenate(ts []NdArray, axis int) NdArray {
	mut newshape := ts[0].shape.clone()
	newshape[axis] = 0
	newshape = assert_shape_off_axis(ts, axis, newshape)
	ret := allocate_cpu(newshape, 'C')
	mut lo := [0].repeat(newshape.len)
	mut hi := newshape.clone()
	hi[axis] = 0
	for t in ts {
		if t.shape[axis] != 0 {
			hi[axis] += t.shape[axis]
			ret.slice_hilo(lo, hi).assign(t)
			lo[axis] = hi[axis]
		}
	}
	return ret
}

// for arrays that are too large, calculate only the leading and trailing
// items along each axis
fn leading_trailing(a NdArray, edgeitems int, lo, hi []int) NdArray {
	axis := lo.len
	if axis == a.ndims {
		return a.slice_hilo(lo, hi)
	}
	if a.shape[axis] > 2 * edgeitems {
		mut flo := lo.clone()
		mut fhi := hi.clone()
		mut slo := lo.clone()
		mut shi := hi.clone()
		flo << 0
		fhi << edgeitems
		slo << a.shape[axis] + -1 * edgeitems
		shi << a.shape[axis]
		f := leading_trailing(a, edgeitems, flo, fhi)
		l := leading_trailing(a, edgeitems, slo, shi)
		return concatenate([f, l], axis)
	} else {
		mut nlo := lo.clone()
		mut nhi := hi.clone()
		nlo << 0
		nhi << a.shape[axis]
		return leading_trailing(a, edgeitems, nlo, nhi)
	}
}

// extends a line to contain new elements
fn extend_line(s, line, word string, line_width int, next_line_prefix string) []string {
	mut sn := s
	mut ln := line
	needs_wrap := ln.len + word.len > line_width
	if needs_wrap {
		sn += ln + '\n'
		ln = next_line_prefix
	}
	ln += word
	return [sn, ln]
}

// recursively print an ndarray
fn recursor(a NdArray, index []int, hanging_indent string, curr_width int, summary_insert string, edge_items int, separator string, max_len int) string {
	axis := index.len
	axes_left := a.ndims - axis
	if axes_left == 0 {
		return rjust(format_float(a.get(index), false), max_len)
	}
	next_hanging_indent := hanging_indent + ' '
	next_width := curr_width - 1
	a_len := a.shape[axis]
	show_summary := (summary_insert.len > 0) && (2 * edge_items < a_len)
	mut leading_items := 0
	mut trailing_items := a_len
	if show_summary {
		leading_items = edge_items
		trailing_items = edge_items
	}
	mut s := ''
	if axes_left == 1 {
		elem_width := curr_width - 2
		mut line := hanging_indent
		mut lii := 0
		for lii < leading_items {
			mut nidx := index.clone()
			nidx << lii
			word := recursor(a, nidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			ret := extend_line(s, line, word, elem_width, hanging_indent)
			s = ret[0]
			line = ret[1]
			line += separator
			lii++
		}
		if show_summary {
			ret := extend_line(s, line, summary_insert, elem_width, hanging_indent)
			s = ret[0]
			line = ret[1]
			line += separator
		}
		mut tii := trailing_items
		for tii >= 2 {
			mut tidx := index.clone()
			tidx << -1 * tii
			word := recursor(a, tidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			ret := extend_line(s, line, word, elem_width, hanging_indent)
			s = ret[0]
			line = ret[1]
			line += separator
			tii--
		}
		mut lidx := index.clone()
		lidx << -1
		word := recursor(a, lidx, next_hanging_indent, next_width, summary_insert, edge_items,
			separator, max_len)
		ret := extend_line(s, line, word, elem_width, hanging_indent)
		s = ret[0]
		line = ret[1]
		s += line
	} else {
		s = ''
		rem := axes_left - 1
		mut line_sep := separator
		line_sep += '\n'.repeat(rem)
		mut lii := 0
		for lii < leading_items {
			mut nidx := index.clone()
			nidx << lii
			nested := recursor(a, nidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			lii++
			s += hanging_indent + nested + line_sep
		}
		if show_summary {
			s += hanging_indent + summary_insert + ', \n'
		}
		mut tii := trailing_items
		for tii >= 2 {
			mut tidx := index.clone()
			tidx << -1 * tii
			nested := recursor(a, tidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			s += hanging_indent + nested + line_sep
			tii--
		}
		mut lidx := index.clone()
		lidx << -1
		nested := recursor(a, lidx, next_hanging_indent, next_width, summary_insert, edge_items,
			separator, max_len)
		s += hanging_indent + nested
	}
	return '[' + s[hanging_indent.len..] + ']'
}

// format an array, array2string is just a wrapper around this
fn format_array(a NdArray, line_width int, next_line_prefix, separator string, edge_items int, summary_insert string, max_len int) string {
	return recursor(a, [], next_line_prefix, line_width, summary_insert, edge_items, separator,
		max_len)
}

// public method for printing arrays, if custom behavior is needed
pub fn array2string(a NdArray, separator, prefix string) string {
	if a.shape.len == 0 {
		return '[]'
	}
	mut summary_insert := ''
	mut data := a
	if a.size > 1000 {
		summary_insert = '...'
		data = leading_trailing(a, 3, [], [])
	}
	max_len := max_str_len(data)
	mut next_line_prefix := ''
	next_line_prefix += ' '.repeat(prefix.len)
	return format_array(a, 75, next_line_prefix, separator, 3, summary_insert, max_len)
}

// formats a floating point value to be "pretty"
fn format_float(v f64, notation bool) string {
	if notation {
		return v.strsci(3)
	} else {
		unsafe {
			buf := malloc(8 * 5 + 1)
// TODO
			C.sprintf(charptr(buf), '%g', v)
			return tos(buf, vstrlen(buf))
		}
	}
}

// finds the max string length of an ndarray
fn max_str_len(a NdArray) int {
	mut mx := 0
	for iter := a.iter(); !iter.done; iter.next() {
		val := format_float(*iter.ptr, false)
		if val.len > mx {
			mx = val.len
		}
	}
	return mx
}

// adjusts a string to be aligned with one side
// of the output
fn rjust(s string, n int) string {
	diff := n - s.len
	if diff > 0 {
		return ' '.repeat(diff) + s
	} else {
		return s
	}
}
