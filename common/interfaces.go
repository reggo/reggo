package common

import "github.com/gonum/matrix/mat64"

type Rower interface {
	Row([]float64, int) []float64
}

type RowMatrix interface {
	mat64.Matrix
	Rower
}

type MutableRowMatrix interface {
	Rower
	mat64.Mutable
	SetRow(int, []float64) int
}

// See package reggo for description. This is here to avoid circular imports
type Predictor interface {
	Predict(input, output []float64) ([]float64, error)
	PredictBatch(inputs RowMatrix, outputs MutableRowMatrix) (MutableRowMatrix, error)
	InputDim() int
	OutputDim() int
}
