// Implementation of feed-forward neural network

package nnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/reggo/reggo/common"
	predHelp "github.com/reggo/reggo/common/predict"
	"github.com/reggo/reggo/train"
)

func init() {
	common.Register(&Net{})
}

// Net is a simple feed-forward neural net
type Net struct {
	inputDim           int
	outputDim          int
	totalNumParameters int

	grainSize int

	neurons    [][]Neuron
	parameters [][][]float64
}

type netMarshaler struct {
	InputDim           int
	OutputDim          int
	TotalNumParameters int
	Neurons            [][]common.InterfaceMarshaler
	Parameters         [][][]float64
}

func (n *Net) marshal() netMarshaler {
	m := netMarshaler{
		InputDim:           n.inputDim,
		OutputDim:          n.outputDim,
		TotalNumParameters: n.totalNumParameters,
		Parameters:         n.parameters,
	}
	m.Neurons = make([][]common.InterfaceMarshaler, len(n.neurons))
	for i, layer := range n.neurons {
		m.Neurons[i] = make([]common.InterfaceMarshaler, len(layer))
		for j, neur := range layer {
			m.Neurons[i][j] = common.InterfaceMarshaler{neur}
		}
	}
	return m
}

func (n *Net) unmarshal(marshaler netMarshaler) error {
	n.inputDim = marshaler.InputDim
	n.outputDim = marshaler.OutputDim
	n.totalNumParameters = marshaler.TotalNumParameters
	n.parameters = marshaler.Parameters

	if len(n.parameters) == 0 {
		if n.inputDim != 0 {
			return errors.New("No parameters provided")
		}
	}

	// Check that the total number of parameters is correct
	var sum int
	for i := range n.parameters {
		for j := range n.parameters[i] {
			sum += len(n.parameters[i][j])
		}
	}
	if sum != n.totalNumParameters {
		return fmt.Errorf("Sum of the parameters does not match total num parameters")
	}

	n.neurons = make([][]Neuron, len(marshaler.Neurons))
	for i := range n.neurons {
		n.neurons[i] = make([]Neuron, len(marshaler.Neurons[i]))
	}

	// Check that the sizes of neurons and parameters match
	if len(n.neurons) != len(n.parameters) {
		return errors.New("Number of layers in parameters and neurons doesn't match")
	}

	for i := range n.neurons {
		if len(n.neurons[i]) != len(n.parameters[i]) {
			return fmt.Errorf("Number of neurons in layer %d does not match for the parameters and neurons. %d in parameters, %d in neurons", i, len(n.parameters[i]), len(n.neurons[i]))
		}
	}

	// Check that they are all neurons and that the number of parameters is correct

	for i := range marshaler.Neurons {
		for j, iface := range marshaler.Neurons[i] {
			// First, check that the interface marshaler is actually a Neuron
			neur, ok := iface.I.(Neuron)
			if !ok {
				return fmt.Errorf("The %d item of layer %d does not implement Neuron")
			}
			// Check that there is the correct number of parameters
			var np int
			if i == 0 {
				np = neur.NumParameters(n.inputDim)
			} else {
				np = neur.NumParameters(len(n.neurons[i-1]))
			}
			if np != len(n.parameters[i][j]) {
				return parametermismatch(i, j, len(n.parameters[i][j]), np)
			}
			n.neurons[i][j] = neur
		}
	}
	return nil
}

func parametermismatch(i, j, t, np int) error {
	return fmt.Errorf("Number of parameters for neuron %d, %d does not match. There were %d parameters, but the Neuron expects %d parameters", i, j, t, np)
}

func (n *Net) MarshalJSON() ([]byte, error) {
	// TODO: Add some error checking. See if it's a common type hint that they
	// need to marshal the predictor.
	marshal := n.marshal()
	return json.Marshal(marshal)
}

func (n *Net) UnmarshalJSON(data []byte) error {
	var marshal netMarshaler
	err := json.Unmarshal(data, &marshal)
	if err != nil {
		return err
	}
	err = n.unmarshal(marshal)
	if err != nil {
		return err
	}
	n.setGrainSize()
	return nil
}

// InputDim returns the number of inputs expected by the net
func (n *Net) InputDim() int {
	return n.inputDim
}

// InputDim returns the number of outputs expected by Sink
func (n *Net) OutputDim() int {
	return n.outputDim
}

func (n *Net) Predict(input []float64, output []float64) ([]float64, error) {
	if len(input) != n.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if output == nil {
		output = make([]float64, n.outputDim)
	} else {
		if len(output) != n.outputDim {
			return nil, errors.New("output dimension mismatch")
		}
	}
	prevOutput, tmpOutput := newPredictMemory(n.neurons)
	predict(input, n.neurons, n.parameters, prevOutput, tmpOutput, output)
	return output, nil
}

func (n *Net) PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error) {
	batch := batchPredictor{
		neurons:    n.neurons,
		parameters: n.parameters,
		inputDim:   n.InputDim(),
		outputDim:  n.OutputDim(),
	}
	return predHelp.BatchPredict(batch, inputs, outputs, n.inputDim, n.outputDim, n.grainSize)
}

func (n *Net) setGrainSize() {
	// The number of floating point operations is roughly equal to the number of
	// parameters, plus there is some overhead per neuron per function call and
	// some overhead per layer in function calls

	// Numbers determined unscientifically by using benchmark results. Definitely
	// dependent on many things, but these are probably good enough
	neuronOverhead := 70 // WAG, relative to one parameter
	layerOverhead := 200 // relative to one parameter

	var nNeurons int
	for _, layer := range n.neurons {
		nNeurons += len(layer)
	}

	nOps := n.totalNumParameters + nNeurons*neuronOverhead + layerOverhead*len(n.neurons)

	// We want each batch to take around 100µs
	// https://groups.google.com/forum/#!searchin/golang-nuts/Data$20parallelism$20with$20go$20routines/golang-nuts/-9LdBZoAIrk/2ayBvi0U0mQJ

	// Something like "nanoseconds per effective parameter"
	// Determined non-scientifically from benchmarks. This is definitely architecture
	// dependent, but maybe not relative to the overhead of the parallel loop
	c := 0.7

	grainSize := int(math.Ceil(100000 / (c * float64(nOps))))
	if grainSize < 1 {
		grainSize = 1 // This shouldn't happen, but maybe for a REALLY large net. Better safe than sorry
	}

	n.grainSize = grainSize
}

func (n *Net) GrainSize() int {
	return n.grainSize
}

// batchPredictor is a type which implements predHelp.BatchPredictor so that
// the predictions can be computed in parallel
type batchPredictor struct {
	neurons    [][]Neuron
	parameters [][][]float64
	inputDim   int
	outputDim  int
}

// NewPredictor generates the necessary temporary memory and returns a struct to allow
// for concurrent prediction
func (b batchPredictor) NewPredictor() predHelp.Predictor {
	prevOutput, tmpOutput := newPredictMemory(b.neurons)
	return predictor{
		neurons:       b.neurons,
		parameters:    b.parameters,
		tmpOutput:     tmpOutput,
		prevTmpOutput: prevOutput,
		inputDim:      b.inputDim,
		outputDim:     b.outputDim,
	}
}

// predictor is a struct that contains temporary memory to be reused during
// sucessive calls to predict
type predictor struct {
	neurons       [][]Neuron
	parameters    [][][]float64
	tmpOutput     []float64
	prevTmpOutput []float64

	inputDim  int
	outputDim int
}

func (p predictor) Predict(input, output []float64) {
	predict(input, p.neurons, p.parameters, p.prevTmpOutput, p.tmpOutput, output)
}

func (p predictor) InputDim() int {
	return p.inputDim
}

func (p predictor) OutputDim() int {
	return p.outputDim
}

func newPredictMemory(neurons [][]Neuron) (prevOutput, output []float64) {
	// find the largest layer in terms of number of neurons
	max := len(neurons[0])
	for i := 1; i < len(neurons); i++ {
		l := len(neurons[i])
		if l > max {
			max = l
		}
	}
	return make([]float64, max), make([]float64, max)
}

// predict predicts the output from the net. prevOutput and output are
func predict(input []float64, neurons [][]Neuron, parameters [][][]float64, prevTmpOutput, tmpOutput []float64, output []float64) {
	nLayers := len(neurons)

	if nLayers == 1 {
		processLayer(input, neurons[0], parameters[0], output)
		return
	}

	// first layer uses the real input as the input
	tmpOutput = tmpOutput[:len(neurons[0])]
	processLayer(input, neurons[0], parameters[0], tmpOutput)

	// Middle layers use the previous output as input
	for i := 1; i < nLayers-1; i++ {
		// swap the pointers for temporary outputs, and make the new output the correct size
		prevTmpOutput, tmpOutput = tmpOutput, prevTmpOutput
		tmpOutput = tmpOutput[:len(neurons[i])]
		processLayer(prevTmpOutput, neurons[i], parameters[i], tmpOutput)
	}
	// The final layer is the actual output
	processLayer(tmpOutput, neurons[nLayers-1], parameters[nLayers-1], output)
}

func processLayer(input []float64, neurons []Neuron, parameters [][]float64, output []float64) {
	for i, neuron := range neurons {
		combination := neuron.Combine(parameters[i], input)
		output[i] = neuron.Activate(combination)
	}
}

// Trainer is a wrapper for the feed-forward net for training
type Trainer struct {
	*Net
}

// NewSimpleTrainer constructs a trainable feed-forward neural net with the specified sizes and
// tanh neuron activators in the hidden layer. Common choices for the final layer activator
// are activator.Linear for regression and activator.Tanh for classification.
// nLayers is the number of hidden layers. For now, must be at least one.
func NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer int, hiddenLayerActivator, finalLayerActivator Activator) (*Trainer, error) {
	if inputDim <= 0 {
		return nil, errors.New("non-positive input dimension")
	}
	if outputDim <= 0 {
		return nil, errors.New("non-positive output dimension")
	}
	if inputDim <= 0 {
		return nil, errors.New("non-positive number of neurons per layer")
	}
	/*
		if nHiddenLayers < 1 {
			return nil, errors.New("must have at least one hidden layer")
		}
	*/

	// Create the neurons
	// the hidden layers have the same number of neurons as hidden layers
	// final layer has a number of neurons equal to the number of outputs
	neurons := make([][]Neuron, nHiddenLayers+1)
	for i := 0; i < nHiddenLayers; i++ {
		neurons[i] = make([]Neuron, nNeuronsPerLayer)
		for j := 0; j < nNeuronsPerLayer; j++ {
			neurons[i][j] = SumNeuron{Activator: hiddenLayerActivator}
		}
	}
	neurons[nHiddenLayers] = make([]Neuron, outputDim)
	for i := 0; i < outputDim; i++ {
		neurons[nHiddenLayers][i] = SumNeuron{Activator: finalLayerActivator}
	}
	return NewTrainer(inputDim, outputDim, neurons)
}

// NewTrainer creates a new feed-forward neural net with the given layers
func NewTrainer(inputDim, outputDim int, neurons [][]Neuron) (*Trainer, error) {
	if len(neurons) == 0 {
		return nil, errors.New("net: no neurons given")
	}
	for i := range neurons {
		if len(neurons[i]) == 0 {
			return nil, errors.New("net: layer with no neurons")
		}
	}

	// Create the parameters, the number of parameters, and the parameter index
	nLayers := len(neurons)
	parameters := make([][][]float64, nLayers)

	totalNumParameters := 0
	nLayerInputs := inputDim
	for i, layer := range neurons {
		parameters[i] = make([][]float64, len(layer))
		for j, neuron := range layer {
			nParameters := neuron.NumParameters(nLayerInputs)
			parameters[i][j] = make([]float64, nParameters)
			totalNumParameters += nParameters
		}
		nLayerInputs = len(layer)
	}
	net := &Net{
		inputDim:           inputDim,
		outputDim:          outputDim,
		totalNumParameters: totalNumParameters,
		neurons:            neurons,
		parameters:         parameters,
	}
	net.setGrainSize()
	return &Trainer{net}, nil
}

// TODO: Replace this with a copy so can modify the trainer after releasing the
// predictor
func (s *Trainer) Predictor() common.Predictor {
	return s.Net
}

// NumFeatures returns the input dimension because a feed-forward neural network
// uses the raw inputs
func (s *Trainer) NumFeatures() int {
	return s.inputDim
}

// NumParameters returns the total number of parameters in all of the neurons of the net
func (s *Trainer) NumParameters() int {
	return s.totalNumParameters
}

func (s *Trainer) Featurize(input, feature []float64) {
	if len(feature) != len(input) {
		panic("feature length mismatch")
	}
	copy(feature, input)
}

func (s *Trainer) RandomizeParameters() {
	for i, layer := range s.neurons {
		for j, neuron := range layer {
			neuron.Randomize(s.parameters[i][j])
		}
	}
}

// Parameters returns a copy of all the parameters as a flattened slice.
// Creates a new slice if nil. Panics if non-nil and incorrect length.
func (s *Trainer) Parameters(p []float64) []float64 {
	if p == nil {
		p = make([]float64, s.NumParameters())
	} else {
		if len(p) != s.NumParameters() {
			panic("nnet: parameter size mismatch")
		}
	}
	getparameters(p, s.parameters)
	return p
}

func getparameters(p []float64, parameters [][][]float64) {
	idx := 0
	for i, layer := range parameters {
		for j, _ := range layer {
			newParams := len(parameters[i][j])
			copy(p[idx:idx+newParams], parameters[i][j])
			idx += newParams
		}
	}
}

// SetParameters copies the input parameter slice into the
// parameters of the trainer. Panics if the length of the input
// does not match that of the number of parameters in the net
func (s *Trainer) SetParameters(p []float64) {
	setparameters(s.parameters, p, s.NumParameters())
}

func setparameters(params [][][]float64, p []float64, nParameters int) {
	if len(p) != nParameters {
		panic("nnet: parameter size mismatch")
	}
	idx := 0
	for _, layer := range params {
		for _, neuron := range layer {
			newParams := len(neuron)
			copy(neuron, p[idx:idx+newParams])
			idx += newParams
		}
	}
}

//
func (s *Trainer) NewFeaturizer() train.Featurizer {
	// featurize can be called in parallel, so just returns self
	return s
}

func (s *Trainer) NewLossDeriver() train.LossDeriver {
	return lossDerivWrapper{
		neurons: s.neurons,
		//parameters: copyParameters(s.parameters),
		parameters:   newPerParameterMemory(s.parameters),
		outputs:      newPerNeuronMemory(s.neurons),
		combinations: newPerNeuronMemory(s.neurons),
		nParameters:  s.NumParameters(),
		dLossDOutput: newPerNeuronMemory(s.neurons),
		dLossDInput:  newPerParameterMemory(s.parameters),
		dLossDParam:  newPerParameterMemory(s.parameters),
	}
}

func copyParameters(p [][][]float64) [][][]float64 {
	n := make([][][]float64, len(p))
	for i, layer := range p {
		n[i] = make([][]float64, len(layer))
		for j, _ := range layer {
			n[i][j] = make([]float64, len(p[i][j]))
			copy(n[i][j], p[i][j])
		}
	}
	return n
}

func newPerParameterMemory(params [][][]float64) [][][]float64 {
	n := make([][][]float64, len(params))
	for i, layer := range params {
		n[i] = make([][]float64, len(layer))
		for j, _ := range layer {
			n[i][j] = make([]float64, len(params[i][j]))
		}
	}
	return n
}

func newPerNeuronMemory(n [][]Neuron) [][]float64 {
	sos := make([][]float64, len(n))
	for i, layer := range n {
		sos[i] = make([]float64, len(layer))
	}
	return sos
}

type lossDerivWrapper struct {
	neurons      [][]Neuron
	parameters   [][][]float64
	outputs      [][]float64
	combinations [][]float64
	nParameters  int
	dLossDOutput [][]float64
	dLossDInput  [][][]float64
	dLossDParam  [][][]float64
}

func (l lossDerivWrapper) Predict(parameters, input, predOutput []float64) {
	setparameters(l.parameters, parameters, l.nParameters)
	cachePredict(input, l.neurons, l.parameters, l.combinations, l.outputs, predOutput)
}

// cachePredict predicts the output while caching the result
func cachePredict(input []float64, neurons [][]Neuron, parameters [][][]float64, combinations, outputs [][]float64, predOutput []float64) {
	nLayers := len(neurons)

	// The first layer uses the true input as the input
	cacheProcessLayer(input, neurons[0], parameters[0], combinations[0], outputs[0])

	// Each other layer uses the outputs of the previous layer as input
	for i := 1; i < nLayers; i++ {
		cacheProcessLayer(outputs[i-1], neurons[i], parameters[i], combinations[i], outputs[i])
	}

	// The final predicted output is the output of the last layer
	copy(predOutput, outputs[nLayers-1])
}

// processLayer computes one layer of the neural net
func cacheProcessLayer(input []float64, neurons []Neuron, parameters [][]float64, combinations, outputs []float64) {
	for i, neuron := range neurons {
		combinations[i] = neuron.Combine(parameters[i], input)
		outputs[i] = neuron.Activate(combinations[i])
	}
}

// TODO: Do I need to do it this way? Can't I just do dProdectionDParameter and then
// the outer code does the multiplication? That seems a lot less confusing. I guess
// that doesn't exploit sparsity at all (dPredDWeight is a matrix)

// Deriv computes the derivatives of the loss function with respect to the parameters.
// input, layers, parameters, dLossDPred, combinations, and outputs are all true inputs to the method.
// dLossDParam is the output of the method
// dLossDOutput and dLossDInput are storage for temporary variables
func (loss lossDerivWrapper) Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64) {
	// Parameters is already set in lossDerivWrapper, and train guarantees that results can be cached

	// For each layer, the following holds
	// dL/dp_{k,i,L} = dL/dout_{i,L} * dout_{i,L}/dcomb_{i,L} * dcomb_{i,L}/dp_{k,i,L}
	// where
	// L is the loss
	// p is the  parameter
	// out is the activation function output
	// comb is the combination function output (i.e. activation input)
	// they are indexed by the kth weight of the ith neuron in the Lth layer.
	// so,
	// dL/dw_{k,i,L} = dLossDWParam,
	// dL/da_{i,L} = dLossDOutput
	// dL/da * da/dS = dLossDCombine
	// dL/da * da/dS * dS/dw = dLossDWeight

	// However, the derivative of the loss with respect to the output of that layer
	// is the sum of the influences of that output on the future outputs
	// This influence depends on the weights
	// Specifically,
	// dL/dout_{i,L-1} = sum_j dL/dcomb_{j, L} * dcomb_{j,L} / dinput_{i,j,L}
	// where input in the ith input
	// note the repetition of the i index in the LHS and the last term of the RHS

	// The derivative of the loss with respect to the ouputs of the last layer
	// is the same as the derivative of the loss function with respect to the
	// predictions (because the outputs of the last layer are the predictions)
	dLossDOutput := loss.dLossDOutput
	nLayers := len(loss.parameters)
	copy(dLossDOutput[nLayers-1], dLossDPred)

	for l := nLayers - 1; l > 0; l-- {
		// Compute dLossDParam and dLossDInput. Inputs to the layer are the outputs of the previous layer
		derivativesLayer(loss.neurons[l], loss.parameters[l], loss.outputs[l-1], loss.combinations[l], loss.outputs[l], loss.dLossDOutput[l], loss.dLossDInput[l], loss.dLossDParam[l])
		for j := range dLossDOutput[l-1] {
			dLossDOutput[l-1][j] = 0
		}
		// Find the derivatives of the outputs for the previous layer
		dInputToDOutput(loss.dLossDInput[l], loss.dLossDOutput[l-1])
	}
	// For the last layer, just need to find the derivative
	derivativesLayer(loss.neurons[0], loss.parameters[0], featurizedInput, loss.combinations[0], loss.outputs[0], loss.dLossDOutput[0], loss.dLossDInput[0], loss.dLossDParam[0])

	// copy the dLossDParameter [][][]float64 to the flat slice
	getparameters(dLossDWeight, loss.dLossDParam)
}

// Find the derivatives of the loss function with respect to the parameters and inputs
// Parameters is all of the parameters of that layer
// Inputs is the input to that layer
// Combinations and outputs are the combinations and outputs for the neurons of that layer
// dLossDOutput is the derivative of the loss with respect to the outputs of that layer
// dLossDParam and dLossDInput are stored in place
func derivativesLayer(neurons []Neuron, parameters [][]float64, inputs []float64, combinations, outputs, dLossDOutput []float64, dLossDInput [][]float64, dLossDParam [][]float64) {
	for i, neuron := range neurons {
		dLossNeuron(neuron, parameters[i], inputs, combinations[i], outputs[i], dLossDOutput[i], dLossDParam[i], dLossDInput[i])
	}
}

func dLossNeuron(n Neuron, parameters []float64, inputs []float64, combination, output, dLossDOutput float64, dLossDParam, dLossDInput []float64) {
	dOutputDCombination := n.DActivateDCombination(combination, output)
	dLossDCombination := dLossDOutput * dOutputDCombination

	// Store DCombineDParameters into dLossDParam
	n.DCombineDParameters(parameters, inputs, combination, dLossDParam)
	// Actual dLossDParam is dLossDCombination * dLossDParam
	for i := range dLossDParam {
		dLossDParam[i] *= dLossDCombination
	}

	// Store DCombineDInput into dLossDInput
	n.DCombineDInput(parameters, inputs, combination, dLossDInput)
	// Actual dLossDInput is dLossDCombine * dCombineDInput
	for i := range dLossDInput {
		dLossDInput[i] *= dLossDCombination
	}
}

// dInputToDOutput changes the derivative of the loss wrt the inputs of the layer to
// next layer wrt the input and changes it to the derivative of the loss wrt the inputs
// of the previous layer
func dInputToDOutput(nextLayerDLossDInput [][]float64, previousLayerDLossDOutput []float64) {
	for i := range previousLayerDLossDOutput {
		previousLayerDLossDOutput[i] = 0
	}
	// derivative of the loss with respect to the outputs is the sum of its derivatives
	// into the next layers
	for i := range previousLayerDLossDOutput {
		for _, neurDLossDInput := range nextLayerDLossDInput {
			previousLayerDLossDOutput[i] += neurDLossDInput[i]
		}
	}
}
