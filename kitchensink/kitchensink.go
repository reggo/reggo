package kitchensink

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/reggo/common"
	predHelp "github.com/reggo/predict"
	"github.com/reggo/train"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func init() {
	common.Register(Sink{})
}

// TODO: Generalize fitting method to allow the space binning thing

// Sink is an implementation of the Kitchen Sinks algorithm by Ali Rahimi
// and Ben Recht. The Sink struct is only used for predicting. In order
// to generate a Sink struct, set up a Trainer and use the provided
// routines in reggo/train.
//
// Please see:
//		Rahimi, Ali, and Benjamin Recht. "Random features for large-scale kernel machines."
//			Advances in neural information processing systems. 2007.
// 		Rahimi, Ali, and Benjamin Recht. "Weighted sums of random kitchen sinks: Replacing
//			minimization with randomization in learning." Advances in neural information
//			processing systems. 2008.
// for more details about the algorithm.
type Sink struct {
	kernel    Kernel
	nFeatures int

	inputDim       int
	outputDim      int
	features       *mat64.Dense // Index is feature number then
	featureWeights *mat64.Dense // Index is feature number then output
	b              []float64    // offsets from feature map
}

// InputDim returns the number of inputs expected by Sink
func (s *Sink) InputDim() int {
	return s.inputDim
}

// InputDim returns the number of outputs expected by Sink
func (s *Sink) OutputDim() int {
	return s.outputDim
}

// Predict returns the output at a given input. Returns nil if the length of the inputs
// does not match the trained number of inputs. The input value is unchanged, but
// will be modified during a call to the method
func (sink *Sink) Predict(input []float64, output []float64) ([]float64, error) {
	if len(input) != sink.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if output == nil {
		output = make([]float64, sink.outputDim)
	} else {
		if len(output) != sink.outputDim {
			return nil, errors.New("output dimension mismatch")
		}
	}
	predict(input, sink.features, sink.b, sink.featureWeights, output)
	return output, nil
}

func (sink *Sink) grainSize() int {
	return 500
}

type sinkMarshal struct {
	Kernel         common.InterfaceMarshaler
	NumFeatures    int
	InputDim       int
	OutputDim      int
	B              []float64
	Features       mat64.RawMatrix
	FeatureWeights mat64.RawMatrix
}

func (sink *Sink) MarshalJSON() ([]byte, error) {
	return json.Marshal(sinkMarshal{
		Kernel:         common.InterfaceMarshaler{sink.kernel},
		NumFeatures:    sink.nFeatures,
		InputDim:       sink.inputDim,
		OutputDim:      sink.outputDim,
		B:              sink.b,
		Features:       sink.features.RawMatrix(),
		FeatureWeights: sink.featureWeights.RawMatrix(),
	})
}

func (sink *Sink) UnmarshalJSON([]byte) error {
	panic("not implemented")
}

// PredictBatch makes a prediction for all of the inputs given in inputs concurrently.
// outputs must either be (nSamples x outputDim), or must be nil. If outputs is nil,
// a new matrix will be created to store the predictions
func (sink *Sink) PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error) {
	batch := batchPredictor{
		features:       sink.features,
		featureWeights: sink.featureWeights,
		b:              sink.b,
	}
	return predHelp.BatchPredict(batch, inputs, outputs, sink.inputDim, sink.outputDim, sink.grainSize())
}

// batchPredictor is a wrapper for BatchPredict to allow parallel predictions
type batchPredictor struct {
	features       *mat64.Dense
	featureWeights *mat64.Dense
	b              []float64
}

// There is no temporary memory involved, so can just return itself
func (b batchPredictor) NewPredictor() predHelp.Predictor {
	return b
}

func (b batchPredictor) Predict(input, output []float64) {
	predict(input, b.features, b.b, b.featureWeights, output)
}

// SinkTrainer is a wrapper around Sink that can modify the contents of the struct for
// training purposes
type Trainer struct {
	*Sink
}

// NewSinkTrainer creates a new SinkTrainer object with the given kernel and dimensions
func NewTrainer(inputDim, outputDim, nFeatures int, kernel Kernel) *Trainer {
	sink := &Sink{
		nFeatures: nFeatures,
		kernel:    kernel,
		inputDim:  inputDim,
		outputDim: outputDim,
	}
	features := mat64.NewDense(sink.nFeatures, inputDim, nil)
	sink.kernel.Generate(sink.nFeatures, inputDim, features)
	sink.features = features
	sink.featureWeights = mat64.NewDense(sink.nFeatures, outputDim, nil)
	b := make([]float64, sink.nFeatures)
	for i := range b {
		b[i] = rand.Float64() * math.Pi * 2
	}
	sink.b = b
	return &Trainer{sink}
}

// Predictor returns the underlying kitchen sink as (a now immutable) predictor.
func (s *Trainer) Predictor() common.Predictor {
	return s.Sink
}

// NumFeatures returns the number of features in the Sink
func (s *Trainer) NumFeatures() int {
	return s.nFeatures
}

// NumParameters returns the number of inputs expected by Sink
func (s *Trainer) NumParameters() int {
	return s.outputDim * s.nFeatures
}

// Linear signifies that the prediction is a linear function of the parameters
func (s *Trainer) Linear() {}

// Convex signifies that the prediction is a convex funciton of the parameters
func (s *Trainer) Convex() {}

// GrainSize gives a hint to the parallel for loops what a good batch size is
func (s *Trainer) GrainSize() int {
	return 500
}

// RandomizeParameters sets the parameters to a random value. Useful for
// initialization for gradient-based training algorithms
func (s *Trainer) RandomizeParameters() {
	rm := s.featureWeights.RawMatrix()
	for i := range rm.Data {
		rm.Data[i] = rand.NormFloat64()
	}
}

// Parameters returns the parameters of the kitchen sink as a single
// slice of values.
func (s *Trainer) Parameters(p []float64) []float64 {
	if p == nil {
		p = make([]float64, s.NumParameters())
	} else {
		if len(p) != s.NumParameters() {
			panic("sink: parameter size mismatch")
		}
	}
	rm := s.featureWeights.RawMatrix()
	copy(p, rm.Data)
	return p
}

// SetParameters sets the parameters to the given value in the same
// order as returned by Parameters(). Will panic if the length of the
// input slice is not the same as the expected number of parameters
func (s *Trainer) SetParameters(p []float64) {
	if len(p) != s.NumParameters() {
		panic("sink: parameter size mismatch")
	}
	rm := s.featureWeights.RawMatrix()
	copy(rm.Data, p)
}

// ComputeZ computes the value of z with the given feature vector and b value.
// Sqrt2OverD = math.Sqrt(2.0 / len(nFeatures))
func computeZ(featurizedInput, feature []float64, b float64, sqrt2OverD float64) float64 {
	dot := floats.Dot(featurizedInput, feature)
	return sqrt2OverD * (math.Cos(dot + b))
}

// wrapper for predict, assumes all inputs are correct
func predict(input []float64, features *mat64.Dense, b []float64, featureWeights *mat64.Dense, output []float64) {
	for i := range output {
		output[i] = 0
	}

	nFeatures, _ := features.Dims()
	_, outputDim := featureWeights.Dims()

	sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))
	//for i, feature := range features {
	for i := 0; i < nFeatures; i++ {
		z := computeZ(input, features.RowView(i), b[i], sqrt2OverD)
		for j := 0; j < outputDim; j++ {
			output[j] += z * featureWeights.At(i, j)
		}
	}
}

func predictFeaturized(featurizedInput []float64, featureWeights *mat64.Dense, output []float64) {
	for i := range output {
		output[i] = 0
	}
	for j, zval := range featurizedInput {
		for i, weight := range featureWeights.RowView(j) {
			output[i] += weight * zval
		}
	}
}

// NewFeaturizer returns a featurizer for use in training routines.
func (s *Trainer) NewFeaturizer() train.Featurizer {
	// The sink featurize method can be called in parallel normally, so
	// nothing is created
	return s
}

// Featurize computes the feature values for the input and stores them in
// place into Featurize
func (sink *Trainer) Featurize(input, feature []float64) {
	sqrt2OverD := math.Sqrt(2.0 / float64(sink.nFeatures))
	for i := range feature {
		feature[i] = computeZ(input, sink.features.RowView(i), sink.b[i], sqrt2OverD)
	}
}

func (s *Trainer) NewLossDeriver() train.LossDeriver {
	return lossDerivWrapper{
		nFeatures: s.nFeatures,
		outputDim: s.outputDim,
	}
}

// DerivSink is a wrapper for training with gradient-based optimization
type lossDerivWrapper struct {
	nFeatures int
	outputDim int
}

func (d lossDerivWrapper) Predict(parameters, featurizedInput, predOutput []float64) {
	featureWeights := mat64.NewDense(d.nFeatures, d.outputDim, parameters)
	predictFeaturized(featurizedInput, featureWeights, predOutput)
}

func (d lossDerivWrapper) Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64) {
	// Form a matrix that has the underlying elements as dLossDWeight so the values are modified in place
	//lossMat := mat64.NewDense(d.s.nFeatures, d.s.outputDim, dLossDWeight)
	deriv(featurizedInput, dLossDPred, dLossDWeight)
}

func deriv(z []float64, dLossDPred []float64, dLossDWeight []float64) {
	// dLossDWeight_ij = \sum_k dLoss/dPred_k * dPred_k / dWeight_j

	// Remember, the parameters are stored in row-major order

	nOutput := len(dLossDPred)
	// The prediction is just weights * z, so dPred_jDWeight_i = z_i
	// dLossDWeight = dLossDPred * dPredDWeight
	for i, zVal := range z {
		for j := 0; j < nOutput; j++ {
			dLossDWeight[i*nOutput+j] = zVal * dLossDPred[j]
		}
	}
}
