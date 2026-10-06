module csv

import encoding.csv as stdcsv
import os
import strconv
import vtl

// CsvReadConfig controls numeric CSV tensor loading.
pub struct CsvReadConfig {
pub:
	delimiter   u8 = `,`
	comment     u8 = `#`
	skip_header bool
	skip_rows   int
	use_cols    []int
	max_rows    int = -1
}

// CsvWriteConfig controls CSV serialization of rank-two tensors.
pub struct CsvWriteConfig {
pub:
	delimiter u8 = `,`
	header    []string
	use_crlf  bool
}

// read parses a numeric CSV file into a rank-two tensor. Quoted fields and
// quoted newlines follow the V standard library CSV reader's behavior.
pub fn read[T](path string, config CsvReadConfig) !&vtl.Tensor[T] {
	if config.skip_rows < 0 {
		return error('csv.read: skip_rows must be non-negative')
	}
	if config.max_rows < -1 {
		return error('csv.read: max_rows must be -1 or non-negative')
	}
	mut content := os.read_file(path)!
	if content.len == 0 {
		return error('csv.read: no numeric data found')
	}
	if content.len > 0 && content[content.len - 1] !in [`\n`, `\r`] {
		content += '\n'
	}
	mut reader := stdcsv.new_reader(content, stdcsv.ReaderConfig{
		delimiter: config.delimiter
		comment:   config.comment
	})
	mut values := []T{}
	mut row_count := 0
	mut column_count := -1
	mut skipped := 0
	header_pending := config.skip_header
	mut header_skipped := false
	for {
		fields := reader.read() or {
			if err.msg() == 'encoding.csv: end of file' {
				break
			}
			return err
		}
		if skipped < config.skip_rows {
			skipped++
			continue
		}
		if header_pending && !header_skipped {
			header_skipped = true
			continue
		}
		if fields.len == 0 || fields.all(it.trim_space() == '') {
			continue
		}
		if config.max_rows >= 0 && row_count >= config.max_rows {
			break
		}
		mut selected := []string{cap: if config.use_cols.len > 0 {
			config.use_cols.len
		} else {
			fields.len
		}}
		if config.use_cols.len > 0 {
			for column in config.use_cols {
				if column < 0 || column >= fields.len {
					return error('csv.read: column index ${column} is out of bounds at row ${row_count}')
				}
				selected << fields[column]
			}
		} else {
			selected = fields
		}
		if column_count >= 0 && selected.len != column_count {
			return error('csv.read: row ${row_count} has ${selected.len} columns; expected ${column_count}')
		}
		column_count = selected.len
		for column, field in selected {
			values << parse_value[T](field) or {
				return error('csv.read: invalid value at row ${row_count}, column ${column}: ${err}')
			}
		}
		row_count++
	}
	if row_count == 0 || column_count <= 0 {
		return error('csv.read: no numeric data found')
	}
	return vtl.from_array[T](values, [row_count, column_count])
}

// write serializes a rank-two tensor to CSV, optionally adding a header row.
pub fn write[T](path string, tensor &vtl.Tensor[T], config CsvWriteConfig) ! {
	if tensor.rank() != 2 {
		return error('csv.write: expected a rank-two tensor, got rank ${tensor.rank()}')
	}
	if config.header.len > 0 && config.header.len != tensor.shape[1] {
		return error('csv.write: header has ${config.header.len} columns; tensor has ${tensor.shape[1]}')
	}
	mut writer := stdcsv.new_writer(stdcsv.WriterConfig{
		delimiter: config.delimiter
		use_crlf:  config.use_crlf
	})
	if config.header.len > 0 {
		writer.write(config.header)!
	}
	mut iterator := tensor.iterator()
	for _ in 0 .. tensor.shape[0] {
		mut record := []string{cap: tensor.shape[1]}
		for _ in 0 .. tensor.shape[1] {
			value, _ := iterator.next() or {
				return error('csv.write: tensor data ended before its declared shape')
			}
			record << value_to_string[T](value)!
		}
		writer.write(record)!
	}
	os.write_file(path, writer.str())!
}

fn parse_value[T](value string) !T {
	clean := value.trim_space()
	$if T is bool {
		match clean.to_lower() {
			'true', '1' {
				return true
			}
			'false', '0' {
				return false
			}
			else {
				return error('expected true, false, 1, or 0')
			}
		}
	} $else $if T is f64 {
		return strconv.atof64(clean)!
	} $else $if T is f32 {
		return f32(strconv.atof64(clean)!)
	} $else $if T is i8 {
		return i8(strconv.parse_int(clean, 10, 8)!)
	} $else $if T is i16 {
		return i16(strconv.parse_int(clean, 10, 16)!)
	} $else $if T is i32 {
		return i32(strconv.parse_int(clean, 10, 32)!)
	} $else $if T is i64 {
		return i64(strconv.parse_int(clean, 10, 64)!)
	} $else $if T is int {
		return strconv.atoi(clean)!
	} $else $if T is u8 {
		return u8(strconv.parse_uint(clean, 10, 8)!)
	} $else $if T is u16 {
		return u16(strconv.parse_uint(clean, 10, 16)!)
	} $else $if T is u32 {
		return u32(strconv.parse_uint(clean, 10, 32)!)
	} $else $if T is u64 {
		return strconv.parse_uint(clean, 10, 64)!
	} $else {
		return error('csv.read: unsupported tensor element type ${T.name}')
	}
}

fn value_to_string[T](value T) !string {
	$if T is bool {
		return if value { 'true' } else { 'false' }
	} $else $if T is f32 || T is f64 || T is i8 || T is i16 || T is i32 || T is i64 || T is int || T is u8 || T is u16 || T is u32 || T is u64 {
		return '${value}'
	} $else {
		return error('csv.write: unsupported tensor element type ${T.name}')
	}
}
