module vtl

fn test_vcl_tensor_add() {
	a := from_2d[f64]([[1.0, 2.0], [3.0, 4.0]]) or { panic(err) }
	b := from_2d[f64]([[10.0, 20.0], [30.0, 40.0]]) or { panic(err) }
	va := a.vcl(storage.VclStorageParams{}) or { panic('vcl transfer failed: ${err}') }
	vb := b.vcl(storage.VclStorageParams{}) or { panic('vcl transfer failed: ${err}') }
	vc := va.add(vb) or { panic('VclTensor.add failed: ${err}') }
	result := vc.cpu() or { panic(err) }
	arr := result.to_array()
	assert arr[0] == 11.0
	assert arr[1] == 22.0
	assert arr[2] == 33.0
	assert arr[3] == 44.0
}

fn test_vcl_tensor_multiply() {
	a := from_1d[f64]([2.0, 3.0, 4.0], TensorData{}) or { panic(err) }
	b := from_1d[f64]([5.0, 6.0, 7.0], TensorData{}) or { panic(err) }
	va := a.vcl(storage.VclStorageParams{}) or { panic(err) }
	vb := b.vcl(storage.VclStorageParams{}) or { panic(err) }
	vc := va.multiply(vb) or { panic('VclTensor.multiply failed: ${err}') }
	result := vc.cpu() or { panic(err) }
	arr := result.to_array()
	assert arr[0] == 10.0
	assert arr[1] == 18.0
	assert arr[2] == 28.0
}

fn test_vcl_tensor_relu() {
	a := from_1d[f64]([-2.0, -1.0, 0.0, 1.0, 2.0], TensorData{}) or { panic(err) }
	va := a.vcl(storage.VclStorageParams{}) or { panic(err) }
	vb := va.relu() or { panic('VclTensor.relu failed: ${err}') }
	result := vb.cpu() or { panic(err) }
	arr := result.to_array()
	assert arr[0] == 0.0
	assert arr[1] == 0.0
	assert arr[2] == 0.0
	assert arr[3] == 1.0
	assert arr[4] == 2.0
}

fn test_vcl_tensor_matmul() {
	a := from_2d[f64]([[1.0, 2.0], [3.0, 4.0]]) or { panic(err) }
	b := from_2d[f64]([[5.0, 6.0], [7.0, 8.0]]) or { panic(err) }
	va := a.vcl(storage.VclStorageParams{}) or { panic(err) }
	vb := b.vcl(storage.VclStorageParams{}) or { panic(err) }
	vc := va.matmul(vb) or { panic('VclTensor.matmul failed: ${err}') }
	result := vc.cpu() or { panic(err) }
	arr := result.to_array()
	assert arr[0] == 19.0
	assert arr[1] == 22.0
	assert arr[2] == 43.0
	assert arr[3] == 50.0
}
