module vtl

fn test_new() {
        init := 1.0
	t := new_tensor<f64>(shape: [3], init: &init)
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[0] == 1.0
}

fn test_from_varray() {
        arr := [1.0, 2.0]
        shape := [2]
	mut t := from_varray<f64>(arr, shape)
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 2
        assert varray[1] == 2.0
}

fn test_zeros() {
	mut t := zeros<f64>([3])
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[1] == 0.0
}

fn test_ones() {
	mut t := ones<f64>([3])
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[1] == 1.0
}

fn test_from_full() {
	mut t := full<f64>([3], 3.0)
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[1] == 3.0
}
