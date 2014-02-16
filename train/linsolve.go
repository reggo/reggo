package train

import (
	"fmt"
	"math"

	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/train/diagonal"

	"github.com/gonum/matrix/mat64"
)

// Creates the features from the inputs. Features must be nSamples x nFeatures or nil
func FeaturizeTrainable(t Trainable, inputs common.RowMatrix, featurizedInputs *mat64.Dense) *mat64.Dense {
	nSamples, nDim := inputs.Dims()
	if featurizedInputs == nil {
		nFeatures := t.NumFeatures()
		featurizedInputs = mat64.NewDense(nSamples, nFeatures, nil)
	}

	rowViewer, isRowViewer := inputs.(mat64.RowViewer)
	var f func(start, end int)
	if isRowViewer {
		f = func(start, end int) {
			featurizer := t.NewFeaturizer()
			for i := start; i < end; i++ {
				featurizer.Featurize(rowViewer.RowView(i), featurizedInputs.RowView(i))
			}
		}
	} else {
		f = func(start, end int) {
			featurizer := t.NewFeaturizer()
			input := make([]float64, nDim)
			for i := start; i < end; i++ {
				inputs.Row(input, i)
				featurizer.Featurize(input, featurizedInputs.RowView(i))
			}
		}
	}

	common.ParallelFor(nSamples, common.GetGrainSize(nSamples, minGrain, maxGrain), f)
	return featurizedInputs
}

// IsLinearRegularizer returns true if the regularizer can be used with LinearSolve
func IsLinearSolveRegularizer(r regularize.Regularizer) bool {
	switch r.(type) {
	case nil:
	case regularize.None:
	default:
		return false
	}
	return true
}

func IsLinearSolveLosser(l loss.Losser) bool {
	switch l.(type) {
	case nil:
	case loss.SquaredDistance:
	default:
		return false
	}
	return true
}

type LinearTrainable interface {
	Trainable
	Linear()
}

// CanLinearSolve returns true if linear solve can be called on the trainable with
// the losser and regularizer
func CanLinearSolve(trainable Trainable, losser loss.Losser, regularizer regularize.Regularizer) bool {
	_, ok := trainable.(LinearTrainable)
	if !ok {
		return false
	}
	if !IsLinearSolveLosser(losser) {
		return false
	}
	if !IsLinearSolveRegularizer(regularizer) {
		return false
	}
	return true
}

// LinearSolve trains a Linear algorithm.
// Assumes inputs and outputs are already scaled
// If features is nil will call featurize
// Will return nil if regularizer is not a linear regularizer
// Is destructive if any of the weights are zero
// Losser is always the two-norm
// Does not set the value of the parameters (in case this is called in parallel with a different routine)
func LinearSolve(linearTrainable LinearTrainable, features *mat64.Dense, inputs, trueOutputs common.RowMatrix,
	weights []float64, regularizer regularize.Regularizer) (parameters []float64) {
	// TODO: Allow tikhonov regularization
	// TODO: Add test for weights
	// TODO: Need to do something about returning a []float64

	if !IsLinearSolveRegularizer(regularizer) {
		return nil
	}

	if features == nil {
		features = FeaturizeTrainable(linearTrainable, inputs, features)
	}

	_, nFeatures := features.Dims()

	var weightedFeatures, weightedOutput *mat64.Dense

	fmt.Println("In linear solve")

	if weights != nil {
		panic("Need functionality to be better. Either banded special case in matrix or do the mulitplication by hand")
		scaledWeight := make([]float64, len(weights))
		for i, weight := range weights {
			scaledWeight[i] = math.Sqrt(weight)
		}
		diagWeight := diagonal.NewDiagonal(len(scaledWeight), scaledWeight)

		nSamples, outputDim := trueOutputs.Dims()
		weightedOutput = mat64.NewDense(nSamples, outputDim, nil)
		weightedFeatures = mat64.NewDense(nSamples, nFeatures, nil)

		weightedOutput.Copy(trueOutputs)
		weightedFeatures.Copy(features)

		// TODO: Replace this with better than mat multiply
		weightedOutput.Mul(diagWeight, weightedOutput)
		weightedFeatures.Mul(diagWeight, weightedFeatures)
	}

	switch regularizer.(type) {
	case nil:
	case regularize.None:
	default:
		panic("Shouldn't be here. Must be error in IsLinearRegularizer")
	}

	if weights == nil {
		parameterMat := mat64.Solve(features, trueOutputs)
		return parameterMat.RawMatrix().Data

	}
	parameterMat := mat64.Solve(weightedFeatures, weightedOutput)

	return parameterMat.RawMatrix().Data
}
