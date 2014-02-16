package regularize

import (
	"math"
	"testing"

	"github.com/gonum/floats"
)

func RegularizerTest(t *testing.T, r Regularizer, name string, parameters []float64, trueLoss float64, trueDeriv []float64) {
	// Test that Loss works
	loss := r.Loss(parameters)
	if math.Abs(loss-trueLoss) > 1e-14 {
		t.Errorf("Loss doesn't match for case %v. Expected: %v, Found: %v", name, trueLoss, loss)
	}
	// Test that LossDeriv works
	derivative := make([]float64, len(trueDeriv))
	lossDeriv := r.LossDeriv(parameters, derivative)

	if math.Abs(lossDeriv-trueLoss) > 1e-14 {
		t.Errorf("Loss doesn't match from LossDeriv for case %v. Expected: %v, Found: %v", name, trueLoss, lossDeriv)
	}
	if !floats.EqualApprox(trueDeriv, derivative, 1e-14) {
		t.Errorf("Derivative doesn't match from LossDeriv for case %v", name)
	}

	for i := range derivative {
		derivative[i] = float64(i)
	}

	lossAddDeriv := r.LossAddDeriv(parameters, derivative)
	if math.Abs(lossAddDeriv-trueLoss) > 1e-14 {
		t.Errorf("Loss doesn't match from LossAddDeriv for case %v. Expected: %v, Found: %v", name, trueLoss, lossAddDeriv)
	}
	for i := range derivative {
		derivative[i] -= float64(i)
	}
	if !floats.EqualApprox(trueDeriv, derivative, 1e-14) {
		t.Errorf("Derivative doesn't match from LossAddDeriv for case %v", name)
	}
}

func TestTwoNorm(t *testing.T) {
	for _, test := range []struct {
		Gamma      float64
		Parameters []float64
		Loss       float64
		Deriv      []float64
		Name       string
	}{
		{
			Gamma:      0.01,
			Parameters: []float64{1, 2},
			Loss:       0.05,
			Deriv:      []float64{0.02, 0.04},
			Name:       "TwoNorm_Basic",
		},
	} {
		RegularizerTest(t, &TwoNorm{Gamma: test.Gamma}, test.Name, test.Parameters, test.Loss, test.Deriv)
	}
}
