package train

import "github.com/reggo/reggo/common"

// TODO: Still would be nice to have a train.Train method which does the smart stuff

// TODO: Move interfaces into their own file
// TODO: Add IsLinear and IsConvex functions

// TODO: Constants should be some function of the flops
const (
	minGrain = 100
	maxGrain = 1000
)

type Featurizer interface {
	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures(). Should not modify input
	Featurize(input, feature []float64)
}

// Linear is a type whose parameters are a linear combination of a set of features
type Trainable interface {
	// NumFeatures returns the number of features
	NumFeatures() int // NumFeatures is how many features the input is transformed into
	InputDim() int
	OutputDim() int
	NumParameters() int             // returns the number of parameters
	Parameters([]float64) []float64 // Puts in place all the parameters into the input, or makes new if it is nil. Panics if wrong size
	SetParameters([]float64)        // Sets the new parameters
	RandomizeParameters()
	NewFeaturizer() Featurizer // Returns a type whose featurize method can be called concurrently
	NewLossDeriver() LossDeriver
	GrainSize() int              // Returns the suggested grain size
	Predictor() common.Predictor // Returns the memory-safe predictor
}

type LossDeriver interface {
	// Gets the current parameters
	//Parameters() []float64

	// Sets the current parameters
	//SetParameters([]float64)

	// Features is either the input or the output from Featurize
	// Deriv will be called after predict so memory may be cached
	Predict(parameters, featurizedInput, predOutput []float64)

	// Deriv computes the derivative of the loss with respect
	// to the weight given the predicted output and the derivative
	// of the loss function with respect to the prediction
	Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64)
}

type BatchPredictor interface {
	NewPredictor() Predictor // Returns a predictor. This exists so that methods can create temporary data if necessary
}

type Predictor interface {
	Predict(input, output []float64)
	InputDim() int
	OutputDim() int
}
