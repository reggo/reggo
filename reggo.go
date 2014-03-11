// Package reggo will eventually provide wrappers when more learning algorithms are implemented and
// the interfaces are fixed. For now they are in flux

package reggo

import (
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

// TODO: Need to move all of the (public) interfaces here. This is the only logical place
// Common (but not public) things should be in common. Can make a comment about trainers
// in Train
// If we want to have somewhere that can do model selection and stuff, that can be in
// selection (or some other sub-package).
// Reggo is based on these interfaces
// 		learning algorithms can be found in algorithms/Xxx
//		Training algorithms can be found in train/Xxx
// Separation between algorithms and learning

/*
type TrainablePreder interface {
	train.Trainable
	Predictor
}
*/

type RowMatrix interface {
	mat64.Matrix
	Row([]float64, int) []float64
}

// A Predictor is a type that can make predictions on data
type Predictor interface {
	// Predict makes a prediction on a single data point, and returns the prediction
	// and any error. If output is nil, a new slice is created and returned. If output
	// is non-nil, the prediction is stored in place and the output is returned. If
	// output is non-nil, and len(output) != outputDim, Predict panics. Similarly,
	// if len(input) != inputDim, Predict panics.
	Predict(input, output []float64) ([]float64, error)

	// PredictBatch predicts a set of list of data where the inputs (and outputs) are stored
	// as rows of a matrix.
	// This may be faster than successive
	// calls to Predict, as the implementer may reuse temporary memory, and may evaluate
	// the predictions concurrently. Like predict, if output is nil, a new Mutable will
	// be created, otherwise the result will be stored in place. Like Predict, if the
	PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error)
}

/*
// Predictor represents an algorithm that can make a prediction.
type Predictor interface {
	Inputs() int
	Outputs() int
	// The predicted value at the input will be stored into the slice
	// at output.
	// Inputs is assumed to have length Inputs(),
	// Outputs is assumed to have length Outputs()
	// Predict may panic if this is not the case.
	Predict(input []float64, output []float64) error
}

// SlicePredictor is a type that can predict a slice of values in parallel
type SlicePredictor interface {
	Predictor
	PredictSlice(input, output [][]float64) error
}

*/

/*
// Trainable is a type that can be trained on a weighted set of inputs
// Probably needs to be some form of setting here
type Trainable interface {
	Train(inputs, outputs mat64.Matrix, weights []float64, losser loss.Losser, regularizer regularize.Regularizer) error
}

/*
// HyperTrainable is a type which also has hyperparameters that can be trained
type HyperTrainable interface {
	TrainHyper(inputs, outputs mat64.Matrix, weights []float64) error
}
*/
/*
// Wrapper around scale
func NewScalePredictor() *ScalePredictor {

}

type ScalePredictor struct {
	SlicePredictor
	InputScaler  scale.Scaler
	OutputScaler scale.Scaler
}

// Wrapper around Train
func NewScaleTrainer() *ScaleTrainer {

}
*/

// TODO: Include a GetTrainable and Get Predictor that take a string
// argument to return a particular type (makes it easy to use flags)

// TODO: Something about a saveable predictor (that can be encoded into json or gob)
// for example
