module main

import vtl.datasets

mut train_ds := datasets.load_imdb(.train, batch_size: 6)?

mut i := 0
for {
	batch := train_ds.next() or { break }
	println('Labels Batch #${i++}')
	println(*batch.labels)
	println('')

	if i == 10 {
		break
	}
}
