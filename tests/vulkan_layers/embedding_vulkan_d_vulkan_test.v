module vulkan_layers

import vtl
import vtl.nn.layers

fn test_embedding_forward_vulkan_basic() {
	// vocab_size=4, embed_dim=3
	// weight rows: [0]=1,2,3  [1]=4,5,6  [2]=7,8,9  [3]=10,11,12
	weight_data := [f32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	weight := vtl.from_1d[f32](weight_data, vtl.TensorData{})!.reshape([4, 3])!

	// input: [1, 4] indices = [0, 2, 1, 3]
	indices_data := [f32(0), 2, 1, 3]
	indices := vtl.from_1d[f32](indices_data, vtl.TensorData{})!.reshape([1, 4])!

	out := layers.embedding_forward_vulkan[f32](indices, weight)!

	assert out.shape == [1, 4, 3]

	// row 0 → [1,2,3]
	assert f32(out.get([0, 0, 0])) == f32(1)
	assert f32(out.get([0, 0, 1])) == f32(2)
	assert f32(out.get([0, 0, 2])) == f32(3)

	// row 2 → [7,8,9]
	assert f32(out.get([0, 1, 0])) == f32(7)
	assert f32(out.get([0, 1, 1])) == f32(8)
	assert f32(out.get([0, 1, 2])) == f32(9)

	// row 1 → [4,5,6]
	assert f32(out.get([0, 2, 0])) == f32(4)
	assert f32(out.get([0, 2, 1])) == f32(5)
	assert f32(out.get([0, 2, 2])) == f32(6)

	// row 3 → [10,11,12]
	assert f32(out.get([0, 3, 0])) == f32(10)
	assert f32(out.get([0, 3, 1])) == f32(11)
	assert f32(out.get([0, 3, 2])) == f32(12)
}

fn test_embedding_forward_vulkan_batch() {
	// vocab_size=3, embed_dim=2, batch=2, seq_len=2
	weight_data := [f32(1), 2, 3, 4, 5, 6]
	weight := vtl.from_1d[f32](weight_data, vtl.TensorData{})!.reshape([3, 2])!

	// indices: [[0,1],[2,0]]
	indices_data := [f32(0), 1, 2, 0]
	indices := vtl.from_1d[f32](indices_data, vtl.TensorData{})!.reshape([2, 2])!

	out := layers.embedding_forward_vulkan[f32](indices, weight)!

	assert out.shape == [2, 2, 2]

	// batch=0, seq=0: idx=0 → [1,2]
	assert f32(out.get([0, 0, 0])) == f32(1)
	assert f32(out.get([0, 0, 1])) == f32(2)

	// batch=0, seq=1: idx=1 → [3,4]
	assert f32(out.get([0, 1, 0])) == f32(3)
	assert f32(out.get([0, 1, 1])) == f32(4)

	// batch=1, seq=0: idx=2 → [5,6]
	assert f32(out.get([1, 0, 0])) == f32(5)
	assert f32(out.get([1, 0, 1])) == f32(6)

	// batch=1, seq=1: idx=0 → [1,2]
	assert f32(out.get([1, 1, 0])) == f32(1)
	assert f32(out.get([1, 1, 1])) == f32(2)
}
