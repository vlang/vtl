module vtl

const (
        max_printable_size = 1000
)

// for arrays that are too large, calculate only the leading and trailing
// items along each axis
fn leading_trailing(t Tensor, edgeitems int, lo []int, hi []int) Tensor {
	axis := lo.len
	if axis == t.rank() {
		return t.slice([lo, hi])
	}
	if t.shape[axis] > 2 * edgeitems {
		mut flo := lo.clone()
		mut fhi := hi.clone()
		mut slo := lo.clone()
		mut shi := hi.clone()
		flo << 0
		fhi << edgeitems
		slo << t.shape[axis] + -1 * edgeitems
		shi << t.shape[axis]
		f := leading_trailing(t, edgeitems, flo, fhi)
		l := leading_trailing(t, edgeitems, slo, shi)
		return concatenate([f, l], axis: axis)
	} else {
		mut nlo := lo.clone()
		mut nhi := hi.clone()
		nlo << 0
		nhi << t.shape[axis]
		return leading_trailing(t, edgeitems, nlo, nhi)
	}
}

// extends a line to contain new elements
fn extend_line(s string, line string, word string, line_width int, next_line_prefix string) []string {
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
fn rprint(t Tensor, index []int, hanging_indent string, curr_width int, summary_insert string, edge_items int, separator string, max_len int) string {
	axis := index.len
	axes_left := t.rank() - axis
	if axes_left == 0 {
		return rjust(format_num(t.get(index), false), max_len)
	}
	next_hanging_indent := hanging_indent + ' '
	next_width := curr_width - 1
	t_len := t.shape[axis]
	show_summary := (summary_insert.len > 0) && (2 * edge_items < t_len)
	mut leading_items := 0
	mut trailing_items := t_len
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
			word := rprint(t, nidx, next_hanging_indent, next_width, summary_insert,
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
			word := rprint(t, tidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			ret := extend_line(s, line, word, elem_width, hanging_indent)
			s = ret[0]
			line = ret[1]
			line += separator
			tii--
		}
		mut lidx := index.clone()
		lidx << -1
		word := rprint(t, lidx, next_hanging_indent, next_width, summary_insert, edge_items,
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
			nested := rprint(t, nidx, next_hanging_indent, next_width, summary_insert,
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
			nested := rprint(t, tidx, next_hanging_indent, next_width, summary_insert,
				edge_items, separator, max_len)
			s += hanging_indent + nested + line_sep
			tii--
		}
		mut lidx := index.clone()
		lidx << -1
		nested := rprint(t, lidx, next_hanging_indent, next_width, summary_insert, edge_items,
			separator, max_len)
		s += hanging_indent + nested
	}
	return '[' + s[hanging_indent.len..] + ']'
}

// format an array, tensor_str is just a wrapper around this
fn format_array(t Tensor, line_width int, next_line_prefix string, separator string, edge_items int, summary_insert string, max_len int) string {
	return rprint(t, [], next_line_prefix, line_width, summary_insert, edge_items, separator, max_len)
}

// public method for printing arrays, if custom behavior is needed
pub fn tensor_str(t Tensor, separator string, prefix string) string {
	if t.shape.len == 0 {
		return '[]'
	}
	mut summary_insert := ''
	mut data := t
	if t.size > max_printable_size {
		summary_insert = '...'
		data = leading_trailing(t, 3, [], [])
	}
	max_len := max_str_len(data)
	mut next_line_prefix := ''
	next_line_prefix += ' '.repeat(prefix.len)
	return format_array(t, 75, next_line_prefix, separator, 3, summary_insert, max_len)
}

// formats a floating point value to be "pretty"
fn format_num(val Num, notation bool) string {
	if notation {
		return val.strsci(3)
	} else {
		unsafe {
			return val.str()
		}
	}
}

// finds the max string length of an ndarray
fn max_str_len(t Tensor) int {
        mut mx := 0
	mut iter := t.iterator()
	for _ in 0 .. t.size {
		val := format_num(iter.next(), false)
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
