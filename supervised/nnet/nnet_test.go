package nnet

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/gonum/floats"

	"github.com/reggo/reggo/common/regtest"
)

var testNets []*Trainer

var nSampleSlice []int = []int{1, 2, 3, 4, 5, 8, 16, 100, 102}

type netIniter struct {
	nHiddenLayers    int
	nNeuronsPerLayer int
	inputDim         int
	outputDim        int
	//nSamples int
	name                string
	finalLayerActivator Activator
}

var netIniters []*netIniter = []*netIniter{
	{
		nHiddenLayers:       0,
		nNeuronsPerLayer:    4,
		inputDim:            12,
		outputDim:           2,
		name:                "No hidden layers",
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       2,
		nNeuronsPerLayer:    5,
		inputDim:            10,
		outputDim:           12,
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       1,
		nNeuronsPerLayer:    5,
		inputDim:            10,
		outputDim:           8,
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       3,
		nNeuronsPerLayer:    5,
		inputDim:            4,
		outputDim:           3,
		finalLayerActivator: Linear{},
	},
}

func init() {
	//runtime.GOMAXPROCS(runtime.NumCPU())
	for i, initer := range netIniters {
		n, err := NewSimpleTrainer(initer.inputDim, initer.outputDim, initer.nHiddenLayers, initer.nNeuronsPerLayer, initer.finalLayerActivator)
		if err != nil {
			panic(err)
		}
		if initer.name == "" {
			initer.name = strconv.Itoa(i)
		}
		n.RandomizeParameters()
		testNets = append(testNets, n)
	}
}

func TestGetAndSetParameters(t *testing.T) {
	for i, initer := range netIniters {
		n := testNets[i]
		regtest.TestGetAndSetParameters(t, n, initer.name)
	}
}

func TestInputOutputDim(t *testing.T) {
	for i, test := range netIniters {
		n := testNets[i]
		regtest.TestInputOutputDim(t, n, test.inputDim, test.outputDim, test.name)
	}
}

const nRandSamp int = 1000

func TestPrivatePredictsMatch(t *testing.T) {
	for i, test := range netIniters {
		for j := 0; j < nRandSamp; j++ {
			n := testNets[i]
			input := make([]float64, test.inputDim)
			floats.Fill(rand.NormFloat64, input)
			outputSimple := make([]float64, test.outputDim)
			floats.Fill(rand.NormFloat64, outputSimple)
			outputCache := make([]float64, test.outputDim)
			floats.Fill(rand.NormFloat64, outputCache)

			// predict using uncached method
			tmp1, tmp2 := newPredictMemory(n.neurons)
			predict(input, n.neurons, n.parameters, tmp1, tmp2, outputSimple)

			// predict using cached method
			combinations := newPerNeuronMemory(n.neurons)
			outputs := newPerNeuronMemory(n.neurons)
			cachePredict(input, n.neurons, n.parameters, combinations, outputs, outputCache)

			if !floats.EqualApprox(outputSimple, outputCache, 1e-14) {
				t.Errorf("test %v: output mismatch between simple and cached predict. Simple: %v, Cached: %v", test.name, outputSimple, outputCache)
				break
			}
		}
	}
}

func TestPublicPredictAndBatch(t *testing.T) {
	for i, test := range netIniters {
		for _, nSamples := range nSampleSlice {
			n := testNets[i]
			inputs := regtest.RandomMat(nSamples, test.inputDim, rand.NormFloat64)
			trueOutputs := regtest.RandomMat(nSamples, test.outputDim, rand.NormFloat64)

			for j := 0; j < nSamples; j++ {
				tmp1, tmp2 := newPredictMemory(n.neurons)
				predict(inputs.RowView(j), n.neurons, n.parameters, tmp1, tmp2, trueOutputs.RowView(j))
				//predict(inputs.RowView(j), s.features, s.b, s.featureWeights, trueOutputs.RowView(j))
			}

			regtest.TestPredictAndBatch(t, n, inputs, trueOutputs, test.name)
		}
	}
}

func TestDeriv(t *testing.T) {
	for i, test := range netIniters {
		for _, nSamples := range nSampleSlice {
			n := testNets[i]
			inputs := regtest.RandomMat(nSamples, test.inputDim, rand.NormFloat64)
			trueOutputs := regtest.RandomMat(nSamples, test.outputDim, rand.NormFloat64)
			regtest.TestDeriv(t, n, inputs, trueOutputs, test.name)
		}
	}
}

func TestJson(t *testing.T) {
	net1, err := NewSimpleTrainer(9, 8, 5, 6, Linear{})
	if err != nil {
		t.Errorf("Error making net", err)
	}
	net1.RandomizeParameters()
	net2, err := NewSimpleTrainer(2, 7, 1, 1, Sigmoid{})
	if err != nil {
		t.Errorf("Error making net", err)
	}
	net2.RandomizeParameters()
	regtest.TestJSON(t, net1.Predictor().(*Net), net2.Predictor().(*Net))
}
