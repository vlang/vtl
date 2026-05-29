module data

import rand
import vsl.la

fn test_dataloader_basic() {
	// Create simple test data: 10 samples, 3 features
	mut x_data := [][]f64{len: 10}
	for i := 0; i < 10; i++ {
		x_data[i] = []f64{len: 3}
		for j := 0; j < 3; j++ {
			x_data[i][j] = f64(i * 3 + j)
		}
	}
	y_labels := [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]!

	x := la.Matrix.deep2[f64](x_data)

	// Test without shuffle
	mut dl := DataLoader.new[f64](x, y_labels, 3, false)

	// Should have 4 batches (ceil(10/3) = 4)
	assert dl.len() == 4

	// Get first batch
	batch_x, batch_y := dl.next()!
	assert batch_x.m == 3
	assert batch_x.n == 3
	assert batch_y.len == 3

	// First batch should be samples 0,1,2
	assert batch_y == [0.0, 1.0, 2.0]

	// Get second batch
	_, batch_y2 := dl.next()!
	assert batch_y2 == [3.0, 4.0, 5.0]

	// Get remaining batches
	_, batch_y3 := dl.next()!
	assert batch_y3 == [6.0, 7.0, 8.0]

	batch_x4, batch_y4 := dl.next()!
	assert batch_y4 == [9.0]
	assert batch_x4.m == 1 // Last batch has only 1 sample

	// No more batches
	dl.next() or {
		assert err.msg().contains('no more batches')
		return
	}
	assert false
}

fn test_dataloader_shuffle() {
	// Create test data: 10 samples, 2 features
	mut x_data := [][]f64{len: 10}
	for i := 0; i < 10; i++ {
		x_data[i] = []f64{len: 2}
		x_data[i][0] = f64(i)
		x_data[i][1] = f64(i) * 2.0
	}
	y_labels := [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]!

	x := la.Matrix.deep2[f64](x_data)

	// Test with shuffle
	rand.seed([u32(42), u32(42)])
	mut dl := DataLoader.new[f64](x, y_labels, 4, true)

	// Collect all samples from first shuffled epoch
	mut all_samples := []int{}
	for {
		_, batch_y := dl.next() or { break }
		for y in batch_y {
			all_samples << int(y)
		}
	}

	// Should have all 10 samples
	assert all_samples.len == 10
}

fn test_dataloader_reset() {
	// Create test data: 6 samples, 2 features
	mut x_data := [][]f64{len: 6}
	for i := 0; i < 6; i++ {
		x_data[i] = []f64{len: 2}
		x_data[i][0] = f64(i)
		x_data[i][1] = f64(i) * 2.0
	}
	y_labels := [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]!

	x := la.Matrix.deep2[f64](x_data)

	mut dl := DataLoader.new[f64](x, y_labels, 2, false)

	// First epoch: get first batch
	_, batch_y := dl.next()!
	assert batch_y == [10.0, 20.0]

	// Reset
	dl.reset()

	// Second epoch: first batch should start from beginning again
	_, batch_y_after_reset := dl.next()!
	assert batch_y_after_reset == [10.0, 20.0]
}

fn test_dataloader_padding() {
	// Create test data: 7 samples, 2 features
	mut x_data := [][]f64{len: 7}
	for i := 0; i < 7; i++ {
		x_data[i] = []f64{len: 2}
		x_data[i][0] = f64(i)
		x_data[i][1] = f64(i) * 2.0
	}
	y_labels := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]!

	x := la.Matrix.deep2[f64](x_data)

	mut dl := DataLoader.new[f64](x, y_labels, 3, false)

	// Should have 3 batches: 3, 3, 1
	batch1_x, batch1_y := dl.next()!
	assert batch1_y == [1.0, 2.0, 3.0]
	assert batch1_x.m == 3

	batch2_x, batch2_y := dl.next()!
	assert batch2_y == [4.0, 5.0, 6.0]
	assert batch2_x.m == 3

	batch3_x, batch3_y := dl.next()!
	assert batch3_y == [7.0]
	assert batch3_x.m == 1
}

fn test_dataset_subset() {
	// Create test data
	x_data := [
		[1.0, 2.0],
		[3.0, 4.0],
		[5.0, 6.0],
		[7.0, 8.0],
		[9.0, 10.0],
	]
	y_labels := [1.0, 2.0, 3.0, 4.0, 5.0]!

	x := la.Matrix.deep2[f64](x_data)

	// Create simple TensorDataset
	mut tensor_ds := TensorDataset.new[f64](x, y_labels)

	// Create subset with indices [0, 2, 4]
	mut subset := Subset.new[f64](tensor_ds, [0, 2, 4])
	assert subset.len() == 3

	// Get sample at subset index 0 (should be original index 0)
	ds_x0, ds_y0 := subset.get(0)!
	assert ds_x0.m == 1
	assert ds_x0.n == 2
	assert ds_x0.get(0, 0) == 1.0
	assert ds_x0.get(0, 1) == 2.0
	assert ds_y0 == [1.0]

	// Get sample at subset index 1 (should be original index 2)
	ds_x1, ds_y1 := subset.get(1)!
	assert ds_x1.m == 1
	assert ds_x1.n == 2
	assert ds_x1.get(0, 0) == 5.0
	assert ds_x1.get(0, 1) == 6.0
	assert ds_y1 == [3.0]

	// Split subset
	train_subset, val_subset := subset.split(0.67)! // ~2 train, ~1 val
	assert train_subset.len() == 2
	assert val_subset.len() == 1
}

fn test_tensor_dataset() {
	// Create test data
	x_data := [
		[1.0, 2.0],
		[3.0, 4.0],
		[5.0, 6.0],
	]
	y_labels := [10.0, 20.0, 30.0]!

	x := la.Matrix.deep2[f64](x_data)

	mut ds := TensorDataset.new[f64](x, y_labels)

	assert ds.len() == 3

	// TensorDataset returns the requested sample and label
	ds_x, ds_y := ds.get(0)!
	assert ds_x.m == 1
	assert ds_x.n == 2
	assert ds_x.get(0, 0) == 1.0
	assert ds_x.get(0, 1) == 2.0
	assert ds_y == [10.0]
}
