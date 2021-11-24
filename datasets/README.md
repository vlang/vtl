# Datasets

## Mnist

```v
module main

import vtl.datasets

fn main() {
	mut ds := datasets.load_mnist(.test, batch_size: 32) ?
	println(ds)

	for {
		batch := ds.next() or { break }
		println(batch)
	}
}
```
