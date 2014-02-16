package polynomial

import (
	"errors"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"

	"github.com/reggo/reggo/common"
	predHelp "github.com/reggo/reggo/predict"
	"github.com/reggo/reggo/train"
)

const (
	minGrain = 100
	maxGrain = 1000
)

// Fits a polynomial with all of the dimensions independent, i.e.
// f(x) = β_0 + β_1,1 x_1 + β_1,2 x_2 + ... + β_1_n x_n
//            + β_2,1 x_1^2 + β_2,2 x_2^2 + ...
type Independent struct {
	order          []int // power of the polynomial is all the dimensions
	nFeatures      int
	inputDim       int
	outputDim      int
	featureWeights *mat64.Dense // Index is feature number then output
}

func (s *Independent) InputDim() int {
	return s.inputDim
}

func (s *Independent) OutputDim() int {
	return s.outputDim
}

func (s *Independent) Predict(input, output []float64) ([]float64, error) {
	if len(input) != s.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if output == nil {
		output = make([]float64, s.outputDim)
	} else {
		if len(output) != s.outputDim {
			return nil, errors.New("output dimension mismatch")
		}
	}
	featurizedInput := make([]float64, s.nFeatures)
	independentFeaturize(input, s.order, featurizedInput)

	outmat := mat64.NewDense(1, s.outputDim, output)
	inmat := mat64.NewDense(1, s.nFeatures, featurizedInput)

	outmat.Mul(inmat, s.featureWeights)
	return outmat.RawMatrix().Data, nil
}

func (s *Independent) grainSize() int {
	return 50
}

func (s *Independent) PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error) {
	batch := batchPredictor{
		featureWeights: s.featureWeights,
		outputDim:      s.outputDim,
		nFeatures:      s.nFeatures,
		order:          s.order,
	}
	return predHelp.BatchPredict(batch, inputs, outputs, s.inputDim, s.outputDim, s.grainSize())
}

type batchPredictor struct {
	featureWeights *mat64.Dense
	outputDim      int
	nFeatures      int
	order          []int
}

func (b batchPredictor) NewPredictor() predHelp.Predictor {
	return predictor{
		featureWeights:  b.featureWeights,
		inputMat:        mat64.NewDense(1, b.nFeatures, nil),
		outputMat:       mat64.NewDense(1, b.outputDim, nil),
		featurizedInput: make([]float64, b.nFeatures),
		order:           b.order,
	}
}

type predictor struct {
	featureWeights  *mat64.Dense
	inputMat        *mat64.Dense
	outputMat       *mat64.Dense
	featurizedInput []float64
	order           []int
}

// predict uses the temporary memory as it is computed sequentially
func (p predictor) Predict(input, output []float64) {
	independentFeaturize(input, order, featurizedInput)
	predictFeaturized(featurizedInput, output, p.featureWeights, p.inMat, p.outMat)
}

func independentFeaturize(input []float64, order []int, featurizedInput []float64) {
	// First entry is one
	featurizedInput[0] = 1.0
	count = 1
	// For every other input, the input number is that entry to a certain power
	for i, val := range input {
		for j := 1; i < order[i]; j++ {
			featurizedInput[count] = math.Pow(val, float64(j))
		}
	}
}

// predictFeaturized multiplies the featureWeights by the featurized input and stores the value. It assumes
// that inMat and outMat already have the correct shape, but will replace the data in them
func predictFeaturized(featurizedInput []float64, output []float64, featureWeights *mat64.Dense, inMat *mat64.Dense, outMat *mat64.Dense) {
	rm := inMat.RawMatrix()
	rmin.Data = featurizedInput
	inMat.LoadRawMatrix(rmin)

	rm = outMat.RawMatrix()
	rm.Data = outMat
	outMat.LoadRawMatrix(rmin)

	// Multiply the feature weights by the featurized input ond store
	outMat.Mul(inMat, featureWeights)
}

type IndependentTrainer struct {
	*Independent
}

func NewIndependentTrainer(inputDim, outputDim int, orders []int) *IndependentTrainer {
	if inputDim <= 0 {
		panic("non-positive input dimension")
	}
	if outputDim <= 0 {
		panic("non-positive output dimension")
	}
	if len(orders) != inputDim {
		panic("length of orders doesn't match inputDim")
	}
	for i, dim := range inputDim {
		panic("negative input dim")
	}

	ind := IndependentTrainer{
		&Independent{
			order:     orders,
			inputDim:  inputDim,
			outputDim: outputDim,
		},
	}
	// Count up the number of features. Number is 1 (for the constant term) plus one for every
	// order
	nFeatures := 1
	for _, val := range orders {
		nFeatures += val
	}
	ind.nFeatures = nFeatures
	featureWeights := mat64.NewDense(nFeatures, outputDim, nil)
	ind.featureWeights = featureWeights
	return
}

func (s *IndependentTrainer) NumFeatures() int {
	return s.nFeatures
}

func (t *Trainer) NumParameters() int {
	// nParameters is the number of features times the output dimension
	return t.nFeatures * t.outputDim
}

// Linear declares that the fitter is linear in the parameters
func (s *IndependentTrainer) Linear() {}

// Convex declares that it's convex in the parameters
func (s *IndependentTrainer) Convex() {}

// GrainSize gives a hint to the parallel for loops what a good batch size is
func (s *Trainer) GrainSize() int {
	return 500
}

// RandomizeParameters sets the parameters to a random value. Useful for
// initialization for gradient-based training algorithms
func (s *IndependentTrainer) RandomizeParameters() {
	rm := s.featureWeights.RawMatrix()
	for i := range rm.Data {
		rm.Data[i] = rand.NormFloat64()
	}
}

// Parameters returns the parameters as a single of values. If
// the input is nil a new slice is created, otherwise the value is
// stored in place. Parameters panics if the length of the input
// is anything but those values
func (s *IndependentTrainer) Parameters(p []float64) []float64 {
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
func (s *IndependentTrainer) SetParameters(p []float64) {
	if len(p) != s.NumParameters() {
		panic("sink: parameter size mismatch")
	}
	rm := s.featureWeights.RawMatrix()
	copy(rm.Data, p)
}

func (s *IndependentTrainer) NewFeaturizer() train.Featurizer {
	// featurize method can be called in parallel anyway, so don't need to create anything
	return s
}

func (s *IndependentTrainer) Featurize(input, feature []float64) {
	independentFeaturize(input, s.order, feature)
}

func (s *IndependentTrainer) NewLossDeriver() train.LossDeriver {

}
