package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"

	"github.com/davecheney/profile"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"

	"github.com/reggo/reggo/nnet"
	"github.com/reggo/reggo/train"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU() - 2)

	gopath := os.Getenv("GOPATH")
	path := filepath.Join(gopath, "prof", "github.com", "reggo", "reggo", "nnet")

	nInputs := 10
	nOutputs := 3
	nLayers := 2
	nNeurons := 50
	nSamples := 1000000
	nRuns := 50

	config := &profile.Config{
		CPUProfile:  true,
		ProfilePath: path,
	}

	defer profile.Start(config).Stop()

	net, err := nnet.NewSimpleTrainer(nInputs, nOutputs, nLayers, nNeurons, nnet.Linear{})
	if err != nil {
		log.Fatal(err)
	}

	// Generate some random data
	inputs := mat64.NewDense(nSamples, nInputs, nil)
	outputs := mat64.NewDense(nSamples, nOutputs, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nInputs; j++ {
			inputs.Set(i, j, rand.Float64())
		}
		for j := 0; j < nOutputs; j++ {
			outputs.Set(i, j, rand.Float64())
		}
	}

	// Create trainer
	prob := train.NewBatchGradBased(net, true, inputs, outputs, nil, nil, nil)
	nParameters := net.NumParameters()

	parameters := make([]float64, nParameters)
	derivative := make([]float64, nParameters)

	for i := 0; i < nRuns; i++ {
		net.RandomizeParameters()
		net.Parameters(parameters)
		prob.ObjGrad(parameters, derivative)
		fmt.Println(floats.Sum(derivative))
	}
}
