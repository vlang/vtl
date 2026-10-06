module csv

import os
import vtl

fn test_csv_reads_quoted_numeric_fields_and_selected_columns() ! {
	path := os.join_path(os.temp_dir(), 'vtl_csv_read_${os.getpid()}.csv')
	defer {
		os.rm(path) or {}
	}
	os.write_file(path, '# input\r\nfeature,label\r\n"1.25",2\r\n3.5,4\r\n')!
	features := read[f64](path, CsvReadConfig{
		skip_header: true
		use_cols:    [0]
	})!
	assert features.shape == [2, 1]
	assert features.to_array() == [1.25, 3.5]
	labels := read[i32](path, CsvReadConfig{
		skip_header: true
		use_cols:    [1]
	})!
	assert labels.to_array() == [2, 4]
}

fn test_csv_reads_bool_and_respects_row_limit() ! {
	path := os.join_path(os.temp_dir(), 'vtl_csv_bool_${os.getpid()}.csv')
	defer {
		os.rm(path) or {}
	}
	os.write_file(path, 'true\nfalse\ntrue\n')!
	values := read[bool](path, CsvReadConfig{
		max_rows: 2
	})!
	assert values.shape == [2, 1]
	assert values.to_array() == [true, false]
}

fn test_csv_reads_single_record_without_line_ending() ! {
	path := os.join_path(os.temp_dir(), 'vtl_csv_single_${os.getpid()}.csv')
	defer {
		os.rm(path) or {}
	}
	os.write_file(path, '1.5,2.5')!
	values := read[f64](path, CsvReadConfig{})!
	assert values.shape == [1, 2]
	assert values.to_array() == [1.5, 2.5]
}

fn test_csv_write_round_trip_with_quoted_header() ! {
	path := os.join_path(os.temp_dir(), 'vtl_csv_write_${os.getpid()}.csv')
	defer {
		os.rm(path) or {}
	}
	values := vtl.from_array[f64]([1.25, -2.0, 3.5, 4.0], [2, 2])!
	write(path, values, CsvWriteConfig{
		header:   ['feature,one', 'feature two']
		use_crlf: true
	})!
	loaded := read[f64](path, CsvReadConfig{
		skip_header: true
	})!
	assert loaded.shape == [2, 2]
	assert loaded.to_array() == [1.25, -2.0, 3.5, 4.0]
}

fn test_csv_rejects_ragged_rows_and_wrong_tensor_rank() ! {
	path := os.join_path(os.temp_dir(), 'vtl_csv_invalid_${os.getpid()}.csv')
	defer {
		os.rm(path) or {}
	}
	os.write_file(path, '1,2\n3\n')!
	if _ := read[f64](path, CsvReadConfig{}) {
		assert false, 'ragged CSV rows must be rejected'
	}
	vector := vtl.from_1d[f64]([1.0, 2.0])!
	if _ := write(path, vector, CsvWriteConfig{}) {
		assert false, 'writing a rank-one tensor must be rejected'
	}
}
