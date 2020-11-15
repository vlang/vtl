module vtl

fn test_new() {
        init := 1.0
	t := new_tensor<f64>(shape: [3], init: &init)
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[0] == 1.0
}

fn main() {
        arr := [1.0, 2.0]
        shape := [2]
	mut t := from_varray<f64>(arr, shape)
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 2
        assert varray[1] == 2.0
}
