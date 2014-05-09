package nnet

import (
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func benchmarkPredict(b *testing.B, inputDim, outputDim, nLayers, nNeurons int) {
	// Construct net
	trainer, err := NewSimpleTrainer(inputDim, outputDim, nLayers, nNeurons, Linear{})
	if err != nil {
		panic(err)
	}
	trainer.RandomizeParameters()
	net := trainer.Predictor()

	input := make([]float64, inputDim)
	for i := range input {
		input[i] = rand.NormFloat64()
	}
	output := make([]float64, outputDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Predict(input, output)
	}
}

func benchmarkPredictBatch(b *testing.B, inputDim, outputDim, nLayers, nNeurons, nSamples int) {
	// Construct net
	trainer, err := NewSimpleTrainer(inputDim, outputDim, nLayers, nNeurons, Linear{})
	if err != nil {
		panic(err)
	}
	trainer.RandomizeParameters()
	net := trainer.Predictor()

	input := mat64.NewDense(nSamples, inputDim, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < inputDim; j++ {
			input.Set(i, j, rand.NormFloat64())
		}
	}

	output := mat64.NewDense(nSamples, outputDim, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.PredictBatch(input, output)
	}
}

func BenchmarkPredict_10_1_1_5(t *testing.B) {
	benchmarkPredict(t, 10, 1, 1, 5)
}

func BenchmarkPredict_10_1_1_50(t *testing.B) {
	benchmarkPredict(t, 10, 1, 1, 50)
}

func BenchmarkPredict_1_1_1_50(t *testing.B) {
	benchmarkPredict(t, 1, 1, 1, 50)
}

func BenchmarkPredict_100_1_1_50(t *testing.B) {
	benchmarkPredict(t, 100, 1, 1, 50)
}

func BenchmarkPredict_10_1_1_5000(t *testing.B) {
	benchmarkPredict(t, 10, 1, 1, 5000)
}

func BenchmarkPredict_10_1_5_5(t *testing.B) {
	benchmarkPredict(t, 10, 1, 5, 50)
}

func BenchmarkPredict_10_1_5_50(t *testing.B) {
	benchmarkPredict(t, 10, 1, 5, 50)
}

func BenchmarkPredict_10_1_5_5000(t *testing.B) {
	benchmarkPredict(t, 10, 1, 5, 5000)
}

func BenchmarkPredictBatch_10_1_1_5_10000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 1, 5, 10000)
}

func BenchmarkPredictBatch_10_1_1_50_100(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 1, 50, 100)
}

func BenchmarkPredictBatch_10_1_1_50_10000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 1, 50, 10000)
}

func BenchmarkPredictBatch_10_1_1_50_100000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 1, 50, 100000)
}

func BenchmarkPredictBatch_10_1_5_5000_1000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 5, 5000, 1000)
}
