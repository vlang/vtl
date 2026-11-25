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

fn test_csv_reads_numpy_text_fixture_with_missing_values_and_explicit_dtype() ! {
	path := os.join_path(os.dir(@FILE), 'testdata', 'numpy_text_missing.csv')
	values := read[f64](path, CsvReadConfig{
		delimiter:      `;`
		comment:        `#`
		skip_header:    true
		missing_values: ['nan']
		fill_value:     '-1'
	})!
	assert values.shape == [3, 2]
	assert values.to_array() == [1.25, 2.5, 3.75, -1.0, 5.0, 6.25]

	single_precision := read[f32](path, CsvReadConfig{
		delimiter:      `;`
		comment:        `#`
		skip_header:    true
		missing_values: ['nan']
		fill_value:     '-1'
	})!
	single_precision_values := single_precision.to_array()
	for index, expected in [1.25, 2.5, 3.75, -1.0, 5.0, 6.25] {
		difference := f64(single_precision_values[index]) - expected
		assert difference > -1e-6 && difference < 1e-6
	}
}

fn test_csv_missing_values_require_a_valid_fill_value() ! {
	path := os.join_path(os.dir(@FILE), 'testdata', 'numpy_text_missing.csv')
	if _ := read[f64](path, CsvReadConfig{
		delimiter:      `;`
		skip_header:    true
		missing_values: ['nan']
	}) {
		assert false, 'missing values must not silently become zero'
	}
	if _ := read[f64](path, CsvReadConfig{
		delimiter:      `;`
		skip_header:    true
		missing_values: ['nan']
		fill_value:     'not-a-number'
	}) {
		assert false, 'fill_value must parse as the requested tensor element type'
	}
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
