import vtl

fn main() {
	left := vtl.from_2d([[1, 2], [3, 4]]) or {
		eprintln(err)
		return
	}
	right := vtl.from_2d([[0, 5], [6, 7]]) or {
		eprintln(err)
		return
	}
	product := vtl.kron(left, right) or {
		eprintln(err)
		return
	}
	println('Kronecker product shape: ${product.shape}')
	println(product)
}
