module vtl

fn test_new() {
        init := 1.0
	t := new_tensor<f64>({
		shape: [3]
                init: &init
	})
        varray := tensor_to_varray<f64>(t)
        assert varray.len == 3
        assert varray[0] == 1.0
}
