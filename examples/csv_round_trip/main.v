module main

import os
import vtl
import vtl.csv

fn main() {
	path := os.join_path(os.temp_dir(), 'vtl_csv_round_trip.csv')
	defer {
		os.rm(path) or {}
	}
	values := vtl.from_array[f64]([1.25, 2.5, 3.75, 5.0], [2, 2])!
	csv.write(path, values, csv.CsvWriteConfig{
		header: ['height, cm', 'weight']
	})!
	loaded := csv.read[f64](path, csv.CsvReadConfig{
		skip_header: true
	})!
	println('shape: ${loaded.shape}')
	println('values: ${loaded.to_array()}')
	without_final_newline := os.join_path(os.temp_dir(), 'vtl_csv_no_final_newline.csv')
	defer {
		os.rm(without_final_newline) or {}
	}
	os.write_file(without_final_newline, '6.0,70.5')!
	single := csv.read[f64](without_final_newline, csv.CsvReadConfig{})!
	println('single record: ${single.to_array()}')
}
