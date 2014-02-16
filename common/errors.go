package common

import (
	"errors"
	"fmt"

	"github.com/gonum/matrix/mat64"
)

type DataMismatch struct {
	Input  int
	Output int
	Weight int
}

func (d DataMismatch) Error() string {
	return fmt.Sprintf("reggo: length mismatch. inputs: %v, outputs: %v, weights: %v ", d.Input, d.Output, d.Weight)
}

var InputDimension error = errors.New("reggo: input dimension mismatch")
var OutputLengths error = errors.New("reggo: output dimension mismatch")
var NoData error = errors.New("reggo: nil data")

// VerifyInputs returns true if the number of rows in inputs is not the same
// as the number of rows in outputs and the length of weights. As a special case,
// the length of weights is allowed to be zero.
func VerifyInputs(inputs, outputs mat64.Matrix, weights []float64) error {
	if inputs == nil || outputs == nil {
		return NoData
	}
	nSamples, _ := inputs.Dims()
	nOutputSamples, _ := outputs.Dims()
	nWeights := len(weights)
	if nSamples != nOutputSamples || (nWeights != 0 && nSamples != nWeights) {
		return DataMismatch{
			Input:  nSamples,
			Output: nOutputSamples,
			Weight: nWeights,
		}
	}
	return nil
}
