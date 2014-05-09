// Package regtest contains a bunch of helper functions for testing regression algorithms

package regtest

import (
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"testing"

	"github.com/btracey/opt/multivariate"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"

	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/train"
)

func init() {
	dbw.Register(goblas.Blas{})
	//rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())
}

const (
	throwPanic = true
	fdStep     = 1e-6
	fdTol      = 1e-6
)

func panics(f func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
		}
	}()
	f()
	return
}

func maybe(f func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
			if throwPanic {
				panic(err)
			}
		}
	}()
	f()
	return
}

func RandomMat(r, c int, f func() float64) *mat64.Dense {
	m := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, f())
		}
	}
	return m
}

type ParameterGetterSetter interface {
	NumParameters() int
	Parameters([]float64) []float64
	SetParameters([]float64)
}

func TestGetAndSetParameters(t *testing.T, p ParameterGetterSetter, name string) {

	// Test that we can get parameters from nil
	// TODO: Add panic guard
	var nilParam []float64
	f := func() {
		nilParam = p.Parameters(nil)
	}

	if maybe(f) {
		t.Errorf("%v: Parameters panicked with nil input", name)
		return
	}

	if len(nilParam) != p.NumParameters() {
		t.Errorf("%v: On nil input, incorrect length returned from Parameters()", name)
	}
	nilParamCopy := make([]float64, p.NumParameters())
	copy(nilParamCopy, nilParam)
	nonNilParam := make([]float64, p.NumParameters())
	p.Parameters(nonNilParam)
	if !floats.Equal(nilParam, nonNilParam) {
		t.Errorf("%v: Return from Parameters() with nil argument and non nil argument are different", name)
	}
	for i := range nonNilParam {
		nonNilParam[i] = rand.NormFloat64()
	}
	if !floats.Equal(nilParam, nilParamCopy) {
		t.Errorf("%v: Modifying the return from Parameters modified the underlying parameters", name)
	}
	setParam := make([]float64, p.NumParameters())
	copy(setParam, nonNilParam)
	p.SetParameters(setParam)
	if !floats.Equal(setParam, nonNilParam) {
		t.Errorf("%v: Input slice modified during call to SetParameters", name)
	}

	afterParam := p.Parameters(nil)
	if !floats.Equal(afterParam, setParam) {
		t.Errorf("%v: Set parameters followed by Parameters don't return the same argument", name)
	}

	// Test that there are panics on bad length arguments
	badLength := make([]float64, p.NumParameters()+3)

	f = func() {
		p.Parameters(badLength)
	}
	if !panics(f) {
		t.Errorf("%v: Parameters did not panic given a slice too long", name)
	}
	f = func() {
		p.SetParameters(badLength)
	}
	if !panics(f) {
		t.Errorf("%v: SetParameters did not panic given a slice too long", name)
	}
	if p.NumParameters() == 0 {
		return
	}
	badLength = badLength[:p.NumParameters()-1]
	f = func() {
		p.Parameters(badLength)
	}
	if !panics(f) {
		t.Errorf("%v: Parameters did not panic given a slice too short", name)
	}
	f = func() {
		p.SetParameters(badLength)
	}
	if !panics(f) {
		t.Errorf("%v: SetParameters did not panic given a slice too short", name)
	}
}

type InputOutputer interface {
	InputDim() int
	OutputDim() int
}

func TestInputOutputDim(t *testing.T, io InputOutputer, trueInputDim, trueOutputDim int, name string) {
	inputDim := io.InputDim()
	outputDim := io.OutputDim()
	if inputDim != trueInputDim {
		t.Errorf("%v: Mismatch in input dimension. expected %v, found %v", name, trueInputDim, inputDim)
	}
	if outputDim != trueOutputDim {
		t.Errorf("%v: Mismatch in input dimension. expected %v, found %v", name, trueOutputDim, inputDim)
	}
}

type Predictor interface {
	Predict(input, output []float64) ([]float64, error)
	PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error)
	InputOutputer
}

// TestPredict tests that predict returns the expected value, and that calling predict in parallel
// also works
func TestPredictAndBatch(t *testing.T, p Predictor, inputs, trueOutputs common.RowMatrix, name string) {
	nSamples, inputDim := inputs.Dims()
	if inputDim != p.InputDim() {
		panic("input Dim doesn't match predictor input dim")
	}
	nOutSamples, outputDim := trueOutputs.Dims()
	if outputDim != p.OutputDim() {
		panic("outpuDim doesn't match predictor outputDim")
	}
	if nOutSamples != nSamples {
		panic("inputs and outputs have different number of rows")
	}

	// First, test sequentially
	for i := 0; i < nSamples; i++ {
		trueOut := make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			trueOut[j] = trueOutputs.At(i, j)
		}
		// Predict with nil
		input := make([]float64, inputDim)
		inputCpy := make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			input[j] = inputs.At(i, j)
			inputCpy[j] = inputs.At(i, j)
		}

		out1, err := p.Predict(input, nil)
		if err != nil {
			t.Errorf(name + ": Error predicting with nil output")
			return
		}
		if !floats.Equal(input, inputCpy) {
			t.Errorf("%v: input changed with nil input for row %v", name, i)
			break
		}
		out2 := make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			out2[j] = rand.NormFloat64()
		}

		_, err = p.Predict(input, out2)
		if err != nil {
			t.Errorf("%v: error predicting with non-nil input for row %v", name, i)
			break
		}
		if !floats.Equal(input, inputCpy) {
			t.Errorf("%v: input changed with non-nil input for row %v", name, i)
			break
		}

		if !floats.Equal(out1, out2) {
			t.Errorf(name + ": different answers with nil and non-nil predict ")
			break
		}
		if !floats.EqualApprox(out1, trueOut, 1e-14) {
			t.Errorf("%v: predicted output doesn't match for row %v. Expected %v, found %v", name, i, trueOut, out1)
			break
		}
	}

	// Check that predict errors with bad sized arguments
	badOuput := make([]float64, outputDim+1)
	input := make([]float64, inputDim)
	for i := 0; i < inputDim; i++ {
		input[i] = inputs.At(0, i)
	}
	output := make([]float64, outputDim)
	for i := 0; i < outputDim; i++ {
		output[i] = trueOutputs.At(0, i)
	}

	_, err := p.Predict(input, badOuput)
	if err == nil {
		t.Errorf("Predict did not throw an error with an output too large")
	}
	if outputDim > 1 {
		badOuput := make([]float64, outputDim-1)
		_, err := p.Predict(input, badOuput)
		if err == nil {
			t.Errorf("Predict did not throw an error with an output too small")
		}
	}

	badInput := make([]float64, inputDim+1)
	_, err = p.Predict(badInput, output)
	if err == nil {
		t.Errorf("Predict did not err when input is too large")
	}
	if inputDim > 1 {
		badInput := make([]float64, inputDim-1)
		_, err = p.Predict(badInput, output)
		if err == nil {
			t.Errorf("Predict did not err when input is too small")
		}
	}

	// Now, test batch
	// With non-nil
	inputCpy := &mat64.Dense{}
	inputCpy.Clone(inputs)
	predOutput, err := p.PredictBatch(inputs, nil)
	if err != nil {
		t.Errorf("Error batch predicting: %v", err)
	}
	if !inputCpy.Equals(inputs) {
		t.Errorf("Inputs changed during call to PredictBatch")
	}
	predOutputRows, predOutputCols := predOutput.Dims()
	if predOutputRows != nSamples || predOutputCols != outputDim {
		t.Errorf("Dimension mismatch after predictbatch with nil input")
	}

	outputs := mat64.NewDense(nSamples, outputDim, nil)
	_, err = p.PredictBatch(inputs, outputs)

	pd := predOutput.(*mat64.Dense)
	if !pd.Equals(outputs) {
		t.Errorf("Different outputs from predict batch with nil and non-nil")
	}

	badInputs := mat64.NewDense(nSamples, inputDim+1, nil)
	_, err = p.PredictBatch(badInputs, outputs)
	if err == nil {
		t.Error("PredictBatch did not err when input dim too large")
	}
	badInputs = mat64.NewDense(nSamples+1, inputDim, nil)
	_, err = p.PredictBatch(badInputs, outputs)
	if err == nil {
		t.Errorf("PredictBatch did not err with row mismatch")
	}
	badOuputs := mat64.NewDense(nSamples, outputDim+1, nil)
	_, err = p.PredictBatch(inputs, badOuputs)
	if err == nil {
		t.Errorf("PredictBatch did not err with output dim too large")
	}
}

type DerivTester interface {
	train.Trainable
	//RandomizeParameters()
}

// TestDeriv uses finite difference to test that the prediction from Deriv
// is correct, and tests that computing the loss in parallel works properly
// Only does finite difference for the first nTest to save time
func TestDeriv(t *testing.T, trainable DerivTester, inputs, trueOutputs common.RowMatrix, name string) {

	// Set the parameters to something random
	trainable.RandomizeParameters()

	// Compute the loss and derivative
	losser := loss.SquaredDistance{}
	regularizer := regularize.TwoNorm{}

	batchGrad := train.NewBatchGradBased(trainable, true, inputs, trueOutputs, nil, losser, regularizer)

	derivative := make([]float64, trainable.NumParameters())
	parameters := trainable.Parameters(nil)
	// Don't need to check loss, because if predict is right and losser is right then loss must be correct
	_ = batchGrad.ObjGrad(parameters, derivative)

	fdDerivative := make([]float64, trainable.NumParameters())

	wg := &sync.WaitGroup{}
	wg.Add(trainable.NumParameters())
	for i := 0; i < trainable.NumParameters(); i++ {
		go func(i int) {
			newParameters := make([]float64, trainable.NumParameters())
			tmpDerivative := make([]float64, trainable.NumParameters())
			copy(newParameters, parameters)
			newParameters[i] += fdStep
			loss1 := batchGrad.ObjGrad(newParameters, tmpDerivative)
			newParameters[i] -= 2 * fdStep
			loss2 := batchGrad.ObjGrad(newParameters, tmpDerivative)
			newParameters[i] += fdStep
			fdDerivative[i] = (loss1 - loss2) / (2 * fdStep)
			wg.Done()
		}(i)
	}
	wg.Wait()
	if !floats.EqualApprox(derivative, fdDerivative, 1e-6) {
		t.Errorf("%v: deriv doesn't match: Finite Difference: %v, Analytic: %v", name, fdDerivative, derivative)
	}
}

/*
// this wrapper here is to call gofunopter until there is a better option
type batchWrapper struct {
	*train.BatchGradBased
}

func (bw batchWrapper) ObjGrad(x []float64) (obj float64, grad []float64, err error) {
	grad = make([]float64, bw.Dimension())
	loss := bw.BatchGradBased.ObjGrad(x, grad)
	return loss, grad, nil
}
*/

// TestLinearsolveAndDeriv compares the optimal weights found from gradient-based optimization with those found
// from computing a linear solve
func TestLinearsolveAndDeriv(t *testing.T, linear train.LinearTrainable, inputs, trueOutputs common.RowMatrix, name string) {
	// Compare with no weights

	rows, cols := trueOutputs.Dims()
	predOutLinear := mat64.NewDense(rows, cols, nil)
	parametersLinearSolve := train.LinearSolve(linear, nil, inputs, trueOutputs, nil, nil)

	linear.SetParameters(parametersLinearSolve)
	linear.Predictor().PredictBatch(inputs, predOutLinear)

	//fmt.Println("Pred out linear", predOutLinear)

	linear.RandomizeParameters()
	parameters := linear.Parameters(nil)

	batch := train.NewBatchGradBased(linear, true, inputs, trueOutputs, nil, loss.SquaredDistance{}, regularize.None{})
	problem := batch
	settings := multivariate.DefaultSettings()
	settings.GradAbsTol = 1e-11
	//settings. = 0

	result, err := multivariate.OptimizeGrad(problem, parameters, settings, nil)
	if err != nil {
		t.Errorf("Error training: %v", err)
	}

	parametersDeriv := result.Loc

	deriv := make([]float64, linear.NumParameters())

	loss1 := batch.ObjGrad(parametersDeriv, deriv)

	linear.SetParameters(parametersDeriv)
	predOutDeriv := mat64.NewDense(rows, cols, nil)
	linear.Predictor().PredictBatch(inputs, predOutDeriv)

	linear.RandomizeParameters()
	init2 := linear.Parameters(nil)
	batch2 := train.NewBatchGradBased(linear, true, inputs, trueOutputs, nil, loss.SquaredDistance{}, regularize.None{})
	problem2 := batch2
	result2, err := multivariate.OptimizeGrad(problem2, init2, settings, nil)
	parametersDeriv2 := result2.Loc

	//fmt.Println("starting deriv2 loss")
	deriv2 := make([]float64, linear.NumParameters())
	loss2 := batch2.ObjGrad(parametersDeriv2, deriv2)

	//fmt.Println("starting derivlin loss")
	derivlinear := make([]float64, linear.NumParameters())
	lossLin := batch2.ObjGrad(parametersLinearSolve, derivlinear)

	_ = loss1
	_ = loss2
	_ = lossLin

	/*

		fmt.Println("param deriv 1 =", parametersDeriv)
		fmt.Println("param deriv2  =", parametersDeriv2)
		fmt.Println("linear params =", parametersLinearSolve)

		fmt.Println("deriv1 loss =", loss1)
		fmt.Println("deriv2 loss =", loss2)
		fmt.Println("lin loss    =", lossLin)

		fmt.Println("deriv    =", deriv)
		fmt.Println("deriv2   =", deriv2)
		fmt.Println("linderiv =", derivlinear)

		//fmt.Println("Pred out deriv", predOutDeriv)

	*/

	/*
		for i := 0; i < rows; i++ {
			fmt.Println(predOutLinear.RowView(i), predOutBatch.RowView(i))
		}
	*/

	if !floats.EqualApprox(parametersLinearSolve, parametersDeriv, 1e-8) {
		t.Errorf("Parameters don't match for gradient based and linear solve.")
		//for i := range parametersDeriv {
		//	fmt.Printf("index %v: Deriv = %v, linsolve = %v, diff = %v\n", i, parametersDeriv[i], parametersLinearSolve[i], parametersDeriv[i]-parametersLinearSolve[i])
		//}
	}

}

type Jsoner interface {
	MarshalJSON() ([]byte, error)
	UnmarshalJSON([]byte) error
}

func TestJSON(t *testing.T, jsoner1 Jsoner, jsoner2 Jsoner) {
	b, err := jsoner1.MarshalJSON()
	if err != nil {
		t.Errorf("Error marshaling: ", err)
	}
	err = jsoner2.UnmarshalJSON(b)
	if err != nil {
		t.Errorf("Error unmarshaling", err)
	}

	if !reflect.DeepEqual(jsoner1, jsoner2) {
		t.Errorf("Not equal after json marshal and unmarshal")
	}
}
