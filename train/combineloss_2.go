package train

import (
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
)

var _ optimize.Gradient = &BatchGradient{}

// BatchGradient optimizes the parameters of a Trainable with gradient-based
type BatchGradient struct {
	// Must be set
	Trainable Trainable
	Inputs    common.RowMatrix
	Outputs   common.RowMatrix
	Weights   []float64

	// Will be defaulted
	Workers     int
	Losser      loss.DerivLosser
	Regularizer regularize.Regularizer

	// for caching the value for the optimizer
	lastParams []float64
	lastGrad   []float64
	lastFunc   float64

	nSamples  int
	inputDim  int
	outputDim int
	features  *mat64.Dense
}

func (b *BatchGradient) Init() {
	b.nSamples, b.inputDim = b.Inputs.Dims()
	var nSamples int
	nSamples, b.outputDim = b.Outputs.Dims()
	if nSamples != b.nSamples {
		panic("batchgradient: sample length mismatch")
	}
	if b.Weights != nil {
		if len(b.Weights) != b.nSamples {
			panic("batchgradient: weight length mismatch")
		}
	}
	if nSamples == 0 {
		panic("batchgradient: no sample data")
	}
	if b.inputDim == 0 {
		panic("batchgradient: no input features")
	}
	if b.outputDim == 0 {
		panic("batchgradient: no output features")
	}
	if b.Losser == nil {
		b.Losser = loss.SquaredDistance{}
	}
	if b.Regularizer == nil {
		b.Regularizer = regularize.None{}
	}
	if b.Workers == 0 {
		b.Workers = 1
	}
	nParameters := b.Trainable.NumParameters()
	if nParameters == 0 {
		panic("batchgradient: no parameters in trainable")
	}
	b.lastParams = make([]float64, nParameters)
	b.lastParams[0] = math.NaN()
	b.lastGrad = make([]float64, nParameters)
	b.features = FeaturizeTrainable(b.Trainable, b.Inputs, nil)
}

func (b *BatchGradient) Func(params []float64) float64 {
	if floats.Equal(params, b.lastParams) {
		return b.lastFunc
	}
	b.lastFunc = b.funcGrad(params, b.lastGrad)
	return b.lastFunc
}

func (b *BatchGradient) Grad(params, deriv []float64) {
	if floats.Equal(params, b.lastParams) {
		copy(deriv, b.lastGrad)
		return
	}
	b.lastFunc = b.funcGrad(params, b.lastGrad)
	copy(deriv, b.lastGrad)
}

func (b *BatchGradient) funcGrad(params, deriv []float64) float64 {
	nParameters := len(deriv)

	// Send out all of the work
	done := make(chan result)
	sz := b.nSamples / b.Workers
	sent := 0
	for i := 0; i < b.Workers; i++ {
		outputDim := b.outputDim
		last := sent + sz
		if i == b.Workers-1 {
			last = b.nSamples
		}
		go func(sent, last int) {
			lossDeriver := b.Trainable.NewLossDeriver()
			predOutput := make([]float64, outputDim)
			dLossDPred := make([]float64, outputDim)
			dLossDParam := make([]float64, nParameters)
			outputs := make([]float64, outputDim)
			tmpderiv := make([]float64, nParameters)
			var totalLoss float64
			for i := sent; i < last; i++ {
				lossDeriver.Predict(params, b.features.RawRowView(i), predOutput)
				b.Outputs.Row(outputs, i)
				loss := b.Losser.LossDeriv(predOutput, outputs, dLossDPred)
				if b.Weights == nil {
					totalLoss += loss
				} else {
					totalLoss += b.Weights[i] * loss
				}
				lossDeriver.Deriv(params, b.features.RawRowView(i), predOutput, dLossDPred, dLossDParam)
				if b.Weights != nil {
					floats.Scale(b.Weights[i], dLossDParam)
				}
				floats.Add(tmpderiv, dLossDParam)
			}
			done <- result{totalLoss, tmpderiv}
		}(sent, last)
		sent += sz
	}
	// Collect all the results
	var totalLoss float64
	for i := range deriv {
		deriv[i] = 0
	}
	for i := 0; i < b.Workers; i++ {
		w := <-done
		totalLoss += w.loss
		floats.Add(deriv, w.deriv)
	}
	// Compute the regularizer
	if b.Regularizer != nil {
		tmp := make([]float64, nParameters)
		totalLoss += b.Regularizer.LossDeriv(params, tmp)
		floats.Add(deriv, tmp)
	}
	sumWeights := float64(b.nSamples)
	if b.Weights != nil {
		sumWeights = floats.Sum(b.Weights)
	}
	totalLoss /= sumWeights
	floats.Scale(1/sumWeights, deriv)
	return totalLoss
}

type result struct {
	loss  float64
	deriv []float64
}
