package kitchensink

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/gonum/blas/cblas"
	"github.com/gonum/matrix/mat64"

	"github.com/gonum/floats"
	"github.com/reggo/reggo/regtest"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/train"
)

var (
	sink = &Sink{}
)

type sinkIniter struct {
	nFeatures int
	kernel    Kernel
	inputDim  int
	outputDim int
	nSamples  int
	name      string
}

var testSinks []*Trainer

var sinkIniters []*sinkIniter = []*sinkIniter{

	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  100,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  9,
		name:      "",
	},
	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  101,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  1,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  2,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		nSamples:  1006,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 13,
		kernel:    IsoSqExp{},
		inputDim:  3,
		outputDim: 5,
		nSamples:  100,
	},
	{
		nFeatures: 2,
		kernel:    IsoSqExp{},
		inputDim:  3,
		outputDim: 5,
		nSamples:  100,
	},
	{
		nFeatures: 2,
		kernel:    IsoSqExp{},
		inputDim:  6,
		outputDim: 4,
		nSamples:  100,
	},
	{
		nFeatures: 8,
		kernel:    IsoSqExp{},
		inputDim:  15,
		outputDim: 3,
		nSamples:  100,
	},
	{
		nFeatures: 8,
		kernel:    IsoSqExp{},
		inputDim:  4,
		outputDim: 12,
		nSamples:  100,
	},
}

func init() {
	mat64.Register(cblas.Blas{})
	// Set up all of the test sinks
	for i, initer := range sinkIniters {
		s := NewTrainer(initer.inputDim, initer.outputDim, initer.nFeatures, initer.kernel)
		s.features = randomMat(initer.nFeatures, initer.inputDim)
		s.featureWeights = randomMat(initer.nFeatures, initer.outputDim)
		s.b = randomSlice(initer.nFeatures)
		testSinks = append(testSinks, s)
		if initer.name == "" {
			initer.name = strconv.Itoa(i)
		}
	}

}

func TestGetAndSetParameters(t *testing.T) {
	for i, test := range sinkIniters {
		s := testSinks[i]
		numParameters := s.NumParameters()
		trueNparameters := test.nFeatures * test.outputDim
		if numParameters != trueNparameters {
			t.Errorf("case %v: NumParameter mismatch. expected %v, found %v", test.name, trueNparameters, numParameters)
		}
		regtest.TestGetAndSetParameters(t, s, test.name)
	}
}

func TestInputOutputDim(t *testing.T) {
	for i, test := range sinkIniters {
		s := testSinks[i]
		regtest.TestInputOutputDim(t, s, test.inputDim, test.outputDim, test.name)
	}
}

func TestComputeZ(t *testing.T) {
	for _, test := range []struct {
		x         []float64
		feature   []float64
		b         float64
		z         float64
		nFeatures float64
		name      string
	}{
		{
			name:      "General",
			x:         []float64{2.0, 1.0},
			feature:   []float64{8.1, 6.2},
			b:         0.8943,
			nFeatures: 50,
			z:         -0.07188374176,
		},
	} {
		z := computeZ(test.x, test.feature, test.b, math.Sqrt(2.0/test.nFeatures))
		if floats.EqualWithinAbsOrRel(z, test.z, 1e-14, 1e-14) {
			t.Errorf("z mismatch for case %v. %v expected, %v found", test.name, test.z, z)
		}
	}
}

func flatten(sos [][]float64) *mat64.Dense {
	r := len(sos)
	if r == 0 {
		return mat64.NewDense(0, 0, nil)
	}
	c := len(sos[0])
	mat := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		if len(sos[i]) != c {
			panic("sos not a matrix")
		}
		for j := 0; j < c; j++ {
			mat.Set(i, j, sos[i][j])
		}
	}
	return mat
}

func TestPredictFeaturized(t *testing.T) {
	for _, test := range []struct {
		z              []float64
		featureWeights [][]float64
		output         []float64
		Name           string
	}{
		{
			Name: "General",
			z:    []float64{1, 2, 3},
			featureWeights: [][]float64{
				{3, 4},
				{1, 2},
				{0.5, 0.4},
			},
			output: []float64{6.5, 9.2},
		},
	} {
		zCopy := make([]float64, len(test.z))
		copy(zCopy, test.z)
		fwMat := flatten(test.featureWeights)
		fwMatCopy := &mat64.Dense{}
		fwMatCopy.Clone(fwMat)

		output := make([]float64, len(test.output))

		predictFeaturized(zCopy, fwMat, output)

		// Test that z wasn't changed
		if !floats.Equal(test.z, zCopy) {
			t.Errorf("z changed during call")
		}

		if !floats.EqualApprox(output, test.output, 1e-14) {
			t.Errorf("output doesn't match for test %v. Expected %v, found %v", test.Name, test.output, output)
		}
	}
}

func TestPrivatePredict(t *testing.T) {
	for _, test := range []struct {
		input          []float64
		features       [][]float64
		b              []float64
		featureWeights [][]float64
		Name           string
	}{
		{
			input: []float64{8, 9, 10},
			featureWeights: [][]float64{
				{8, 9},
				{0.4, 0.2},
				{9.8, 1.6},
				{-4, -8},
			},
			features: [][]float64{
				{0.9, 0.8, 0.7},
				{-0.7, 0.2, 15},
				{1.5, 7.8, -2.4},
				{9.7, 9.2, 1.2},
			},
			b:    []float64{0.7, 1.2, 0.2, 0.01234},
			Name: "General",
		},
	} {
		inputCopy := make([]float64, len(test.input))
		copy(inputCopy, test.input)

		fwMat := flatten(test.featureWeights)
		fwMatCopy := &mat64.Dense{}
		fwMatCopy.Clone(fwMat)

		featureMat := flatten(test.features)
		featureMatCopy := &mat64.Dense{}
		featureMatCopy.Clone(featureMat)

		bCopy := make([]float64, len(test.b))
		copy(bCopy, test.b)

		// This test assumes ComputeZ and PredictWithZ work
		nOutput := len(test.featureWeights[0])
		nFeatures := len(test.featureWeights)
		zOutput := make([]float64, nOutput)
		predOutput := make([]float64, nOutput)

		for i := range predOutput {
			predOutput[i] = rand.NormFloat64()
			zOutput[i] = rand.NormFloat64()
		}

		sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))

		z := make([]float64, nFeatures)
		for i := range z {
			z[i] = computeZ(test.input, featureMat.RowView(i), test.b[i], sqrt2OverD)
		}
		predictFeaturized(z, fwMat, zOutput)

		predict(inputCopy, featureMatCopy, bCopy, fwMatCopy, predOutput)

		// Check to make sure nothing changed
		if !floats.Equal(inputCopy, test.input) {
			t.Errorf("input has been modified")
		}
		if !floats.Equal(bCopy, test.b) {
			t.Errorf("b has been modified")
		}
		if !fwMat.Equals(fwMatCopy) {
			t.Errorf("feature weights changed")
		}
		if !featureMat.Equals(featureMatCopy) {
			t.Errorf("features changed")
		}
		if !floats.EqualApprox(zOutput, predOutput, 1e-14) {
			t.Errorf("Prediction doesn't match for case %v. Expected %v, found %v", test.Name, zOutput, predOutput)
		}
	}
}

func TestPredict(t *testing.T) {
	for i, test := range sinkIniters {
		s := testSinks[i]
		inputs := randomMat(test.nSamples, test.inputDim)
		trueOutputs := randomMat(test.nSamples, test.outputDim)

		for j := 0; j < test.nSamples; j++ {
			predict(inputs.RowView(j), s.features, s.b, s.featureWeights, trueOutputs.RowView(j))
		}

		regtest.TestPredictAndBatch(t, s, inputs, trueOutputs, test.name)
	}
}

func randomSlice(l int) []float64 {
	s := make([]float64, l)
	for i := range s {
		s[i] = rand.Float64()
	}
	return s
}

func randomMat(r, c int) *mat64.Dense {
	m := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, rand.NormFloat64())
		}
	}
	return m
}

func TestDeriv(t *testing.T) {
	for i, test := range sinkIniters {
		sink := testSinks[i]
		inputs := randomMat(test.nSamples, test.inputDim)
		trueOutputs := randomMat(test.nSamples, test.outputDim)
		regtest.TestDeriv(t, sink, inputs, trueOutputs, test.name)
	}
}

func TestDerivTrain(t *testing.T) {
	nDim := 1
	nTrain := 16000
	xTrain, yTrain := generateRandomSamples(nTrain, nDim)

	nFeatures := 100

	sigmaSq := 0.01

	kernel := &IsoSqExp{LogScale: math.Log(sigmaSq)}

	sink := NewTrainer(nDim, nDim, nFeatures, kernel)

	regtest.TestLinearsolveAndDeriv(t, sink, xTrain, yTrain, "basicsink")
}

// TODO: Need to update this when we have a training function

func TestQuality(t *testing.T) {
	// On a certain function we know that the prediction is good
	// confirm that it is
	nDim := 2
	nTrain := 160
	xTrain, yTrain := generateRandomSamples(nTrain, nDim)

	nTest := 1000
	xTest, yTest := generateRandomSamples(nTest, nDim)

	nFeatures := 300

	sigmaSq := 0.01

	kernel := &IsoSqExp{LogScale: math.Log(sigmaSq)}

	sink := NewTrainer(nDim, nDim, nFeatures, kernel)

	parameters := train.LinearSolve(sink, nil, xTrain, yTrain, nil, regularize.None{})
	sink.SetParameters(parameters)

	/*
		batchGrad := train.NewBatchGradBased(sink, true, xTrain, yTrain, nil, loss.SquaredDistance{}, regularize.None{})

		derivative := make([]float64, sink.NumParameters())
		batchGrad.ObjGrad(parameters, derivative)
		fmt.Println("Quality derivative")
		//fmt.Println(derivative)
		fmt.Println("sum deriv = ", floats.Sum(derivative))
		sink.RandomizeParameters()
		sink.Parameters(parameters)
		batchGrad.ObjGrad(parameters, derivative)
		fmt.Println("sum deriv 2 = ", floats.Sum(derivative))

	*/

	// Predict on new values
	pred, err := sink.PredictBatch(xTest, nil)
	if err != nil {
		t.Errorf(err.Error())
	}
	for i := 0; i < nTest; i++ {
		for j := 0; j < nDim; j++ {
			diff := pred.At(i, j) - yTest.At(i, j)
			if math.Abs(diff) > 1e-9 {
				t.Errorf("Mismatch sample %v, output %v. Expected %v, Found %v", i, j, yTest.At(i, j), pred.At(i, j))
			}
		}
	}
}

func testfunc(x float64) float64 {
	return math.Sin(x/20) + x*x
}

func generateRandomSamples(n, nDim int) (x, y *mat64.Dense) {
	x = mat64.NewDense(n, nDim, nil)
	y = mat64.NewDense(n, nDim, nil)

	for i := 0; i < n; i++ {
		for j := 0; j < nDim; j++ {
			x.Set(i, j, rand.NormFloat64())
			y.Set(i, j, testfunc(x.At(i, j)))
		}
	}
	return
}

/*
func TestKitchenSink(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())
	// generate data
	mat64.Register(cblas.Blas{})
	nDim := 1
	nTrain := 1600
	xTrain, yTrain := generateRandomSamples(nTrain, nDim)

	nTest := 10000
	xTest, yTest := generateRandomSamples(nTest, nDim)

	nFeatures := 300

	// generate z
	sigmaSq := 0.01

	kernel := &IsoSqExp{LogScale: math.Log(sigmaSq)}

	// Train the struct
	sink := NewSink(nFeatures, kernel)
	sink.Train(xTrain, yTrain, nil, nil, nil)

	// Predict on trained values
	for i := 0; i < nTrain; i++ {
		_, _ = sink.Predict(xTrain.RowView(i), nil)
		//fmt.Println(yTrain.At(i, 0), pred[0], yTrain.At(i, 0)-pred[0])
	}
	fmt.Println()
	// Predict on new values
	pred, err := sink.PredictBatch(xTest, nil)
	if err != nil {
		t.Errorf(err.Error())
	}
	if nTest < 1000 {
		for i := 0; i < nTest; i++ {
			fmt.Println(pred.At(i, 0), yTest.At(i, 0), yTest.At(i, 0)-pred.At(i, 0))
		}
	}

	/*
		// TODO: Test weights
		weights := make([]float64, nTrain)
		for i := range weights {
			weights[i] = rand.Float64()
		}
		sink.Train(xTrain, yTrain, weights, nil, nil)
*/

/*
	for i := range testPred {
		fmt.Println(yTest[i][0], "\t", testPred[i][0], "\t", yTest[i][0]-testPred[i][0])
	}
*/
//}
