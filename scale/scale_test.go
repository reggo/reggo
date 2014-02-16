package scale

import (
	"bytes"
	"encoding/gob"
	"math"
	"reflect"
	"testing"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func flatten(data [][]float64) *mat64.Dense {
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

func testGob(s Scaler, sdecode Scaler, t *testing.T) {
	w := new(bytes.Buffer)
	encoder := gob.NewEncoder(w)
	err := encoder.Encode(s)
	if err != nil {
		t.Error(err)
	}

	b := w.Bytes()
	r := bytes.NewBuffer(b)
	decoder := gob.NewDecoder(r)
	err = decoder.Decode(sdecode)
	if err != nil {
		t.Error(err)
	}
	isequal := reflect.DeepEqual(s, sdecode)
	if !isequal {
		t.Errorf("reflect DeepEqual doesn't match")
	}
}

// TODO: Add in more tests for bad inputs

func testScaling(t *testing.T, u Scaler, data *mat64.Dense, scaledData *mat64.Dense, name string) {
	// Copy data

	r, c := data.Dims()
	origData := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			origData.Set(i, j, data.At(i, j))
		}
	}

	err := ScaleData(u, data)
	if err != nil {
		t.Errorf("Error found in ScaleData for case " + name + ": " + err.Error())
	}
	/*
		for i := range data {
			if !floats.EqualApprox(data[i], scaledData[i], 1e-14) {
				t.Errorf("Improper scaling for case"+name+". Expected: %v, Found: %v", data[i], scaledData[i])
			}
		}
	*/

	if !data.EqualsApprox(scaledData, 1e-14) {
		t.Errorf("Improper scaling for case"+name+". Expected: %v, Found: %v", scaledData, data)
	}

	err = UnscaleData(u, data)
	if err != nil {
		t.Errorf("Error found in UnscaleData for case " + name + ": " + err.Error())
	}

	if !data.EqualsApprox(origData, 1e-14) {
		t.Errorf("Improper unscaling for case"+name+". Expected: %v, Found: %v", origData, data)
	}
}

func testLinear(t *testing.T, kind linearTest) {
	u := &Linear{}

	data := flatten(kind.data)
	err := u.SetScale(data)

	if err != nil {
		if kind.eqDim != true {
			t.Errorf("Error where there shouldn't be for case " + kind.name + ": " + err.Error())
		}
	}
	if !floats.EqualApprox(u.Min, kind.min, 1e-14) {
		t.Errorf("Min doesn't match for case " + kind.name)
	}
	if !floats.EqualApprox(u.Max, kind.max, 1e-14) {
		t.Errorf("Max doesn't match for case " + kind.name)
	}
	scaledData := flatten(kind.scaledData)
	testScaling(t, u, data, scaledData, kind.name)
	u2 := &Linear{}
	testGob(u, u2, t)
}

type linearTest struct {
	data       [][]float64
	scaledData [][]float64
	min        []float64
	max        []float64
	name       string
	eqDim      bool
}

func TestLinear(t *testing.T) {
	cases := []linearTest{
		linearTest{
			data: [][]float64{
				[]float64{1},
				[]float64{2},
				[]float64{-3},
				[]float64{-4},
			},
			scaledData: [][]float64{
				[]float64{5.0 / 6.0},
				[]float64{6.0 / 6.0},
				[]float64{1.0 / 6.0},
				[]float64{0.0 / 6.0},
			},
			min:  []float64{-4},
			max:  []float64{2},
			name: "OneD",
		},
		linearTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 9},
				[]float64{-3, 12},
				[]float64{-4, 15},
			},
			scaledData: [][]float64{
				[]float64{5.0 / 6.0, 0},
				[]float64{6.0 / 6.0, 5.0 / 11},
				[]float64{1.0 / 6.0, 8.0 / 11},
				[]float64{0.0 / 6.0, 1},
			},
			min:  []float64{-4, 4},
			max:  []float64{2, 15},
			name: "TwoD",
		},
		linearTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 4},
				[]float64{-3, 4},
				[]float64{-4, 4},
			},
			scaledData: [][]float64{
				[]float64{5.0 / 6.0, 0.5},
				[]float64{6.0 / 6.0, 0.5},
				[]float64{1.0 / 6.0, 0.5},
				[]float64{0.0 / 6.0, 0.5},
			},
			min:   []float64{-4, 3.5},
			max:   []float64{2, 4.5},
			name:  "EqDim",
			eqDim: true,
		},
	}
	for i := range cases {
		testLinear(t, cases[i])
	}
}

type normalTest struct {
	data       [][]float64
	scaledData [][]float64
	mu         []float64
	sigma      []float64
	name       string
	eqDim      bool
}

func testNormal(t *testing.T, kind normalTest) {
	u := &Normal{}
	data := flatten(kind.data)
	err := u.SetScale(data)

	if err != nil {
		if kind.eqDim != true {
			t.Errorf("Error where there shouldn't be for case " + kind.name + ": " + err.Error())
		}
	}
	if !floats.EqualApprox(u.Mu, kind.mu, 1e-14) {
		t.Errorf("Mu doesn't match for case "+kind.name+". Expected: %v, Found: %v", kind.mu, u.Mu)
	}
	if !floats.EqualApprox(u.Sigma, kind.sigma, 1e-14) {
		t.Errorf("Sigma doesn't match for case "+kind.name+". Expected: %v, Found: %v", kind.sigma, u.Sigma)
	}
	scaledData := flatten(kind.scaledData)
	testScaling(t, u, data, scaledData, kind.name)

	u2 := &Normal{}
	testGob(u, u2, t)
}

func TestNormal(t *testing.T) {
	cases := []normalTest{
		normalTest{
			data: [][]float64{
				[]float64{1},
				[]float64{2},
				[]float64{-3},
				[]float64{-4},
			},
			scaledData: [][]float64{
				[]float64{2 / math.Sqrt(6.5)},
				[]float64{3 / math.Sqrt(6.5)},
				[]float64{-2 / math.Sqrt(6.5)},
				[]float64{-3 / math.Sqrt(6.5)},
			},
			mu:    []float64{-1},
			sigma: []float64{math.Sqrt(6.5)},
			name:  "OneD",
		},

		normalTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 9},
				[]float64{-3, 12},
				[]float64{-4, 15},
			},
			scaledData: [][]float64{
				[]float64{2 / math.Sqrt(6.5), -6 / math.Sqrt(16.5)},
				[]float64{3 / math.Sqrt(6.5), -1 / math.Sqrt(16.5)},
				[]float64{-2 / math.Sqrt(6.5), 2 / math.Sqrt(16.5)},
				[]float64{-3 / math.Sqrt(6.5), 5 / math.Sqrt(16.5)},
			},
			mu:    []float64{-1, 10},
			sigma: []float64{math.Sqrt(6.5), math.Sqrt(16.5)},
			name:  "TwoD",
		},

		normalTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 4},
				[]float64{-3, 4},
				[]float64{-4, 4},
			},
			scaledData: [][]float64{
				[]float64{2 / math.Sqrt(6.5), 0},
				[]float64{3 / math.Sqrt(6.5), 0},
				[]float64{-2 / math.Sqrt(6.5), 0},
				[]float64{-3 / math.Sqrt(6.5), 0},
			},
			mu:    []float64{-1, 4},
			sigma: []float64{math.Sqrt(6.5), 1},
			name:  "EqDim",
			eqDim: true,
		},
	}
	for i := range cases {
		testNormal(t, cases[i])
	}
}

/*
type probabilityTest struct {
	data         [][]float64
	unscaledDist []ProbabilityDistribution
	scaledDist   []ProbabilityDistribution
	unscaledData [][]float64
}

/*
func TestProbability(t *testing.T) {
	cases := []probabilityTest{
		probabilityTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 9},
				[]float64{-3, 12},
				[]float64{-4, 15},
			},
			unscaledDist: []ProbabilityDistribution{dist.Laplace{0, 2}, dist.Exponential{2}},
			scaledDist:   []ProbabilityDistribution{dist.Normal{0, 1}, dist.Normal{0, 1}},
		},
	}

	for i :=

	for i := range cases {
		testNormal(t, cases[i])
	}
}
*/
