package common

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func flatten(data [][]float64) *mat64.Dense {
	if data == nil {
		return mat64.NewDense(0, 0, nil)
	}
	nSamples := len(data)
	nDim := len(data[0])
	mat := mat64.NewDense(nSamples, nDim, nil)
	for i := range data {
		if len(data[i]) != nDim {
			panic("bad flatten")
		}
		for j := range data[i] {
			mat.Set(i, j, data[i][j])
		}
	}
	return mat
}

func TestVerifyInputs(t *testing.T) {
	inputs := [][]float64{
		{3, 4, 5},
		{6, 7, 8},
		{9, 10, 11},
	}
	outputs := [][]float64{
		{1, 2},
		{2, 3},
		{9, 10},
	}
	weights := []float64{
		1,
		1.4,
		3.1,
	}
	var err error

	// Test unequal lengths of input and output
	err = VerifyInputs(flatten(inputs), flatten(outputs), weights)
	if err != nil {
		t.Errorf("Error with proper input")
	}
	// Test nil weights allowed
	err = VerifyInputs(flatten(inputs), flatten(outputs), nil)

	if err != nil {
		t.Errorf("Error with nil weights")
	}

	for _, test := range []struct {
		Name                  string
		Input, Output, Weight int
		inputs                [][]float64
		outputs               [][]float64
		weights               []float64
	}{
		{
			Name:   "NilInput",
			Input:  0,
			Output: 3,
			Weight: 3,
			inputs: nil,
			/*
				inputs: [][]float64{
					{3, 4, 5},
					{6, 7, 8},
					{9, 10, 11},
			*/
			outputs: [][]float64{
				{1, 2},
				{2, 3},
				{9, 10},
			},
			weights: []float64{
				1,
				1.4,
				3.1,
			},
		},
		{
			Name:   "ShortInput",
			Input:  2,
			Output: 3,
			Weight: 3,

			inputs: [][]float64{
				{3, 4, 5},
				{6, 7, 8},
			},
			outputs: [][]float64{
				{1, 2},
				{2, 3},
				{9, 10},
			},
			weights: []float64{
				1,
				1.4,
				3.1,
			},
		},
		{
			Name:   "NilOutput",
			Input:  3,
			Output: 0,
			Weight: 3,
			inputs: [][]float64{
				{3, 4, 5},
				{6, 7, 8},
				{9, 10, 11},
			},
			outputs: nil,
			weights: []float64{
				1,
				1.4,
				3.1,
			},
		},
		{
			Name:   "ShortOutput",
			Input:  3,
			Output: 2,
			Weight: 3,

			inputs: [][]float64{
				{3, 4, 5},
				{6, 7, 8},
				{9, 10, 11},
			},
			outputs: [][]float64{
				{1, 2},
				{2, 3},
			},
			weights: []float64{
				1,
				1.4,
				3.1,
			},
		},
		{
			Name:   "ShortWeights",
			Input:  3,
			Output: 3,
			Weight: 2,

			inputs: [][]float64{
				{3, 4, 5},
				{6, 7, 8},
				{9, 10, 11},
			},
			outputs: [][]float64{
				{1, 2},
				{2, 3},
				{9, 10},
			},
			weights: []float64{
				1,
				1.4,
			},
		},
	} {

		inputs := flatten(test.inputs)
		outputs := flatten(test.outputs)

		err = VerifyInputs(inputs, outputs, test.weights)
		misErr, ok := err.(DataMismatch)
		if !ok {
			t.Errorf("%v: Mismatch error not returned with bad inputs", test.Name)
		}
		if misErr.Input != test.Input {
			t.Errorf("%v: incorrect input length", test.Name)
		}
		if misErr.Output != test.Output {
			t.Errorf("%v: incorrect output length")
		}
		if misErr.Weight != test.Weight {
			t.Errorf("%v: incorrect weight length")
		}
	}

	// Verify no data
	err = VerifyInputs(nil, nil, nil)
	if err != NoData {
		t.Errorf("NoData error not returned on no data")
	}
}
