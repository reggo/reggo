// package for guassian processes

package gp

import (
	"errors"
	"github.com/gonum/matrix/mat64"

	"github.com/reggo/kernel"
	"github.com/reggo/reggo/common"
)

// TODO: Need to decide if predictors are safe given modifications of the trainer
// Good uses for both. Probably better to copy, as they feel like different entities,
// so it is surprising that they aren't. Probably need to provide both functionalities

// Lots of to-dos. Make some version of Matrix rather than

type Gp struct {
	kernel.Kernel
	Implicit bool // Should the kernel matrix be stored or computed on the fly (not yet supported)

	inputDim int

	nData int

	inputData  []float64 // Row major data array to make growing the matrix easier
	outputData []float64 // Row major data array to make growing the matrix easier

	kernelData []float64 // Row major data array. This will be "viewed" to get the actual kernel matrix

	inputs  *mat64.Dense
	outputs *mat64.Dense

	kernelMatrix mat64.Matrix // Matrix and not dense so could eventually support implicit
}

// Predict returns the output at a given input. Returns nil if the length of the inputs
// does not match the trained number of inputs. The input value is unchanged, but
// will be modified during a call to the method
func (gp *Gp) Predict(input, output []float64) ([]float64, error) {
	if len(input) != gp.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if output == nil {
		output = make([]float64, sink.outputDim)
	} else {
		if len(output) != sink.outputDim {
			return nil, errors.New("output dimension mismatch")
		}
	}
	predict(input, output)
	return output, nil
}

//
func (gp *Gp) PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error) {

}

// InputDim returns the number of inputs expected
func (gp *Gp) InputDim() int {
	return s.inputDim
}

// InputDim returns the number of outputs expected
func (gp *Gp) OutputDim() int {
	return s.outputDim
}

// VariancePredict returns the prediction and the variance at that point
func (gp *Gp) VariancePredict(input, output []float64, variance []float64) (
	[]float64, []float64, error) {

}

//
func (gp *Gp) VariancePredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix, covariances common.MutableRowMatrix) (
	common.MutableRowMatrix, common.MutableRowMatrix, error) {

}

func (gp *Gp) predict(input, output []float64) {

}

type Trainer struct {
	Gp
}

// Add adds a new point to the gaussian process
func (gp *Trainer) Add(newInput []float64, newOutput []float64) {

	gp.nData++
	// See if we need to allocate new memory
	var inputAtCap bool
	if len(gp.inputData) == cap(gp.inputData) {
		inputAtCap = true
	}
	/*
		var outputAtCap bool
		if len(gp.outputData) == cap(gp.outputData) {
			outputAtCap = true
		}
	*/

	gp.inputData = append(gp.inputData, newInput)
	gp.outputData = append(gp.outputData, newOutput)

	// If we had to allocate memory, allocate new memory for the kernel matrix
	if gp.Implicit {
		// If it's implicit, just need to update matrix size, because the kernel
		// is computed on the fly
		//gp.kernelMat =
		panic("not coded")
	}
	var newKernelMatrix *mat64.Dense
	if inputAtCap {
		oldKernelMatrix := gp.kernelMatrix
		// If we had to allocate new memory for the inputs, then need to expand
		// the size of the matrix as well
		newKernelData := make([]float64, cap(gp.inputData)*cap(gp.inputData))

		panic("Need to use raw matrix")
		//newKernelMatrix = mat64.NewDense(gp.nData, gp.nData, newKernelData)

		// Copy the old kernel data into the new one. View and newKernelMatrix share
		// the same underlying array
		view := &mat64.Dense{}
		view.View(newKernelMatrix, 0, 0, gp.nData-1, gp.nData-1)
		view.Copy(oldKernelMatrix)

		gp.kernelData = newKernelData
	} else {
		// We aren't at capacity, so just need to increase the size
		newKernelMatrix = mat64.NewDense(nData, nData, gp.kernelData)
	}
	// Set the new values of the kernel matrix
	for i := 0; i < nData; i++ {
		oldInput := gp.inputData[i*gp.inputDim : (i+1)*gp.inputDim]
		ker := gp.Kernel(oldData, newInput)
		newKernelMatrix.Set(i, gp.nData, ker)
		newKernelMatrix.Set(gp.nData, i, ker)
	}
	gp.kernelMatrix = newKernelMatrix
}

// AddBatch adds a set of points to the gaussian process fit
func (gp *Trainer) AddBatch(newInput mat64.Matrix, newOutput mat64.Matrix) {

}
