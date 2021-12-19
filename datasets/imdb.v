module datasets

import vtl

pub const (
	imdb_folder_name = 'aclImdb'
	imdb_file_name   = '${imdb_folder_name}_v1.tar.gz'
	imdb_base_url    = 'http://ai.stanford.edu/~amaas/data/sentiment'
)

pub struct ImdbDataset {
pub:
	@type      DatasetType
	batch_size int
}

pub struct ImdbBatch {
pub:
	features &vtl.Tensor<f32>
	labels   &vtl.Tensor<int>
}

pub struct ImdbDatasetConfig {
	batch_size int = 32
}

pub fn load_imdb(set_type DatasetType, data ImdbDatasetConfig) ?&ImdbDataset {
	paths := download_dataset(
		dataset: 'imdb'
		baseurl: datasets.imdb_base_url
		extract: true
		tar: true
		urls_names: {
			'/$datasets.imdb_file_name': datasets.imdb_folder_name
		}
	) ?

	return &ImdbDataset{
		@type: set_type
		batch_size: data.batch_size
	}
}
