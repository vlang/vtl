import vtl
import vtl.stats

fn main() {
	samples := vtl.from_1d([1, 2, 3, 4])!
	population := stats.variance(samples, stats.VarianceData{})!
	sample := stats.variance(samples, stats.VarianceData{
		ddof: 1
	})!
	deviation := stats.std(samples, stats.VarianceData{})!
	eprintln('population variance: ${population}')
	eprintln('sample variance: ${sample}')
	eprintln('population standard deviation: ${deviation}')
}
