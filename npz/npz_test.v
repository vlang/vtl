module npz

import os
import vtl

fn test_npz_round_trip_named_arrays_and_member_listing() ! {
	path := os.join_path(os.temp_dir(), 'vtl_npz_round_trip.npz')
	defer {
		os.rm(path) or {}
	}
	arrays := {
		'weights': vtl.from_array[f64]([1.5, -2.0, 3.25, 4.5], [2, 2])!
		'bias':    vtl.from_1d[f64]([0.25, -0.5])!
	}
	write(path, arrays)!
	assert members(path)! == ['bias', 'weights']
	weights := read[f64](path, 'weights')!
	bias := read[f64](path, 'bias.npy')!
	assert weights.shape == [2, 2]
	assert weights.to_array() == [1.5, -2.0, 3.25, 4.5]
	assert bias.to_array() == [0.25, -0.5]
}

fn test_npz_reads_numpy_compressed_archive_with_mixed_dtypes() ! {
	path := os.join_path(os.dir(@FILE), 'testdata', 'numpy_compressed.npz')
	weights := read[f64](path, 'weights')!
	labels := read[i32](path, 'labels')!
	mask := read[bool](path, 'mask')!
	assert weights.shape == [2, 2]
	assert weights.to_array() == [1.5, -2.0, 3.25, 4.5]
	assert labels.to_array() == [i32(2), 0, 1]
	assert mask.to_array() == [true, false]
}

fn test_npz_writes_and_reads_mixed_dtypes() ! {
	path := os.join_path(os.temp_dir(), 'vtl_npz_mixed_round_trip.npz')
	defer {
		os.rm(path) or {}
	}
	arrays := {
		'weights': array[f64](vtl.from_array[f64]([1.5, -2.0], [2])!)
		'labels':  array[i32](vtl.from_1d[i32]([2, 0, 1])!)
		'mask':    array[bool](vtl.from_1d[bool]([true, false])!)
	}
	write_arrays(path, arrays)!
	assert members(path)! == ['labels', 'mask', 'weights']
	assert read[f64](path, 'weights')!.to_array() == [1.5, -2.0]
	assert read[i32](path, 'labels')!.to_array() == [i32(2), 0, 1]
	assert read[bool](path, 'mask')!.to_array() == [true, false]
}

fn test_npz_rejects_missing_members_and_path_names() {
	path := os.join_path(os.temp_dir(), 'vtl_npz_reject_names.npz')
	defer {
		os.rm(path) or {}
	}
	write(path, {
		'values': vtl.from_1d[f64]([1.0, 2.0])!
	}) or { panic(err) }
	if _ := read[f64](path, 'missing') {
		assert false, 'read must reject a missing member'
	}
	if _ := read[f64](path, '../outside') {
		assert false, 'read must reject path traversal names'
	}
	if _ := write(path, {
		'folder/array': vtl.from_1d[f64]([3.0])!
	}) {
		assert false, 'write must reject member paths'
	}
	if _ := write_arrays(path, {
		'values':     array[f64](vtl.from_1d[f64]([3.0])!)
		'values.npy': array[i32](vtl.from_1d[i32]([4])!)
	}) {
		assert false, 'write_arrays must reject duplicate normalized names'
	}
}
