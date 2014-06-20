package train

import (
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"

	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"

	"fmt"
)

var _ = fmt.Println

// BatchGradBased is a wrapper for training a trainable with
// a fixed set of samples
// TODO: Maybe want to move Featurize out of batch grad (or let pass in features as an input)
type BatchGradBased struct {
	t           Trainable
	losser      loss.DerivLosser
	regularizer regularize.Regularizer

	inputDim    int
	outputDim   int
	nTrain      int
	nParameters int
	grainSize   int

	inputs  common.RowMatrix
	outputs common.RowMatrix

	features *mat64.Dense

	lossDerivFunc func(start, end int, c chan lossDerivStruct, parameters []float64)
}

type lossDerivStruct struct {
	loss  float64
	deriv []float64
}

// NewBatchGradBased creates a new batch grad based with the given inputs
func NewBatchGradBased(trainable Trainable, cacheFeatures bool, inputs, outputs common.RowMatrix, weights []float64, losser loss.DerivLosser, regularizer regularize.Regularizer) *BatchGradBased {
	var features *mat64.Dense
	if cacheFeatures {
		features = FeaturizeTrainable(trainable, inputs, nil)
	}

	// TODO: Add in error checking

	if losser == nil {
		losser = loss.SquaredDistance{}
	}
	if regularizer == nil {
		regularizer = regularize.None{}
	}

	if weights != nil {
		// TODO: Fix weights
		panic("non-nil weights")
	}

	nTrain, outputDim := outputs.Dims()
	_, inputDim := inputs.Dims()
	g := &BatchGradBased{
		t:           trainable,
		inputs:      inputs,
		outputs:     outputs,
		features:    features,
		losser:      losser,
		regularizer: regularizer,
		nTrain:      nTrain,
		outputDim:   outputDim,
		inputDim:    inputDim,
		nParameters: trainable.NumParameters(),
		grainSize:   trainable.GrainSize(),
	}

	// TODO: Add in row viewer stuff
	// TODO: Create a different function for computing just the loss
	//inputRowViewer, ok := inputs.(mat64.RowViewer)
	//outputRowViewer, ok := outputs.(mat64.RowViewer)

	// TODO: Move this to its own function
	var f func(start, end int, c chan lossDerivStruct, parameters []float64)

	switch {
	default:
		panic("Shouldn't be here")
	case cacheFeatures:
		f = func(start, end int, c chan lossDerivStruct, parameters []float64) {
			lossDeriver := g.t.NewLossDeriver()
			prediction := make([]float64, g.outputDim)
			dLossDPred := make([]float64, g.outputDim)
			dLossDWeight := make([]float64, g.nParameters)
			totalDLossDWeight := make([]float64, g.nParameters)
			var loss float64
			output := make([]float64, g.outputDim)
			for i := start; i < end; i++ {
				// Compute the prediction
				lossDeriver.Predict(parameters, g.features.RowView(i), prediction)
				// Compute the loss

				g.outputs.Row(output, i)
				loss += g.losser.LossDeriv(prediction, output, dLossDPred)
				// Compute the derivative
				lossDeriver.Deriv(parameters, g.features.RowView(i), prediction, dLossDPred, dLossDWeight)

				floats.Add(totalDLossDWeight, dLossDWeight)
			}
			// Send the value back on the channel
			c <- lossDerivStruct{
				loss:  loss,
				deriv: totalDLossDWeight,
			}
		}
	case !cacheFeatures:
		f = func(start, end int, c chan lossDerivStruct, parameters []float64) {
			lossDeriver := g.t.NewLossDeriver()
			prediction := make([]float64, g.outputDim)
			dLossDPred := make([]float64, g.outputDim)
			dLossDWeight := make([]float64, g.nParameters)
			totalDLossDWeight := make([]float64, g.nParameters)
			var loss float64
			output := make([]float64, g.outputDim)

			input := make([]float64, g.inputDim)

			features := make([]float64, g.t.NumFeatures())

			featurizer := g.t.NewFeaturizer()
			for i := start; i < end; i++ {

				g.inputs.Row(input, i)

				featurizer.Featurize(input, features)

				// Compute the prediction
				lossDeriver.Predict(parameters, features, prediction)
				// Compute the loss
				g.outputs.Row(output, i)

				loss += g.losser.LossDeriv(prediction, output, dLossDPred)

				// Compute the derivative
				lossDeriver.Deriv(parameters, features, prediction, dLossDPred, dLossDWeight)

				// Add to the total derivative
				floats.Add(totalDLossDWeight, dLossDWeight)

				// Send the value back on the channel
				c <- lossDerivStruct{
					loss:  loss,
					deriv: totalDLossDWeight,
				}
			}
		}
	}

	g.lossDerivFunc = f

	return g
}

// Dimension returns the dimension of the optimization problem
func (g *BatchGradBased) Dimension() int {
	return g.t.NumParameters()
}

// ObjDeriv computes the objective value and stores the derivative in place
func (g *BatchGradBased) ObjGrad(parameters []float64, derivative []float64) (loss float64) {
	c := make(chan lossDerivStruct, 10)

	// Set the channel for parallel for
	f := func(start, end int) {
		g.lossDerivFunc(start, end, c, parameters)
	}

	go func() {
		wg := &sync.WaitGroup{}
		// Compute the losses and the derivatives all in parallel
		wg.Add(2)
		go func() {
			common.ParallelFor(g.nTrain, g.grainSize, f)
			wg.Done()
		}()
		// Compute the regularization
		go func() {
			deriv := make([]float64, g.nParameters)
			loss := g.regularizer.LossDeriv(parameters, deriv)
			//fmt.Println("regularizer loss = ", loss)
			//fmt.Println("regularizer deriv = ", deriv)
			c <- lossDerivStruct{
				loss:  loss,
				deriv: deriv,
			}
			wg.Done()
		}()
		// Wait for all of the results to be sent on the channel
		wg.Wait()
		// Close the channel
		close(c)
	}()
	// zero the derivative
	for i := range derivative {
		derivative[i] = 0
	}

	// Range over the channel, incrementing the loss and derivative
	// as they come in
	for l := range c {
		loss += l.loss
		floats.Add(derivative, l.deriv)
	}
	//fmt.Println("nTrain", g.nTrain)
	//fmt.Println("final deriv", derivative)
	// Normalize by the number of training samples
	loss /= float64(g.nTrain)
	floats.Scale(1/float64(g.nTrain), derivative)

	return loss
}
