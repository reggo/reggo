package reggo

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

// Trainable is a type that can be trained on a weighted set of inputs
type Trainable interface {
	Train(inputs, outputs [][]float64, weights []float64) error
}

// HyperTrainable is a type which also has hyperparameters that can be trained
type HyperTrainable interface {
	TrainHyper(inputs, outputs [][]float64, weights []float64) error
}

// TODO: Include a GetTrainable and Get Predictor that take a string
// argument to return a particular type (makes it easy to use flags)

// TODO: Something about a saveable predictor (that can be encoded into json or gob)
// for example
