package nnet

import (
	"encoding/gob"
	"encoding/json"
	"math"
	"math/rand"

	"github.com/reggo/reggo/common"

	"fmt"
)

func init() {
	gob.Register(SumNeuron{Activator: LinearTanh{}})
	common.Register(SumNeuron{})
}

var (
	TanhNeuron       SumNeuron = SumNeuron{Activator: Tanh{}}
	LinearTanhNeuron SumNeuron = SumNeuron{Activator: LinearTanh{}}
	LinearNeuron     SumNeuron = SumNeuron{Activator: Linear{}}
	SigmoidNeuron    SumNeuron = SumNeuron{Activator: Sigmoid{}}
)

// Neuron doesn't provide own memory, just a definition. Net interfaces with parameters directly
type Neuron interface {
	NumParameters(nInputs int) int // How many parameters as a function of the number of inputs

	Activate(combination float64) (output float64)
	Combine(parameters, inputs []float64) (combination float64)

	Randomize(parameters []float64) // Set to a random initial condition for doing random restarts

	DActivateDCombination(combination, output float64) (derivative float64)
	DCombineDParameters(params []float64, inputs []float64, combination float64, deriv []float64)
	DCombineDInput(params []float64, inputs []float64, combination float64, deriv []float64)

	// Shouldn't need to add json.Marshaler because layer can do it
}

// A sum neuron takes a weighted sum of all the inputs and pipes them through an activator function
type SumNeuron struct {
	Activator
}

// Activate function comes from activator

// NParameters returns the number of parameters
func (s SumNeuron) NumParameters(nInputs int) int {
	return nInputs + 1
}

// Combine takes a weighted sum of the inputs with the weights set by parameters
// The last element of parameters is the bias term, so len(parameters) = len(inputs) + 1
func (s SumNeuron) Combine(parameters []float64, inputs []float64) (combination float64) {
	for i, val := range inputs {
		combination += parameters[i] * val
	}
	combination += parameters[len(parameters)-1]
	return
}

// Randomize sets the parameters to a random initial condition
func (s SumNeuron) Randomize(parameters []float64) {
	for i := range parameters {
		parameters[i] = rand.NormFloat64() * math.Pow(float64(len(parameters)), -0.5)
	}
}

// DActivateDCombination comes from activator

func (s SumNeuron) DCombineDParameters(params []float64, inputs []float64, combination float64, deriv []float64) {
	// The derivative of the function with respect to the parameters (in this case, the weights), is just
	// the value of the input, and 1 for the bias term
	for i, val := range inputs {
		deriv[i] = val
	}
	deriv[len(deriv)-1] = 1
}

// DCombineDInput Finds the derivative of the combination with respect to the ith input
// The derivative of the combination with respect to the input is the value of the weight
func (s SumNeuron) DCombineDInput(params []float64, inputs []float64, combination float64, deriv []float64) {
	for i := range inputs {
		deriv[i] = params[i]
	}
	// This intentionally doesn't loop over all of the parameters, as the last parameter is the bias term
}

/*
func (s SumNeuron) D2CombineD2Input(params, inputs []float64, combination float64, deriv []float64, hessian *mat64.Dense) {
	for
}
*/

type activatorMarshaler struct {
	Activator common.InterfaceMarshaler
}

func (s SumNeuron) MarshalJSON() (b []byte, err error) {
	c := &common.InterfaceMarshaler{s.Activator}
	b, err = json.Marshal(c)
	return b, err
}

func (s *SumNeuron) UnmarshalJSON(data []byte) error {
	v := common.InterfaceMarshaler{}
	err := json.Unmarshal(data, &v)
	if err != nil {
		return fmt.Errorf("nnet/neuron/unmarshaljson: error unmarshaling data: " + err.Error())
	}
	s.Activator = v.I.(Activator)
	return nil
}
