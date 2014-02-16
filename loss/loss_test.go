package loss

import (
	"math"
	"testing"

	"github.com/gonum/floats"
	"github.com/reggo/reggo/common"
)

const (
	FDStep = 10E-6
	FDTol  = 10E-8
	TOL    = 1E-14
)

func finiteDifferenceLosser(losser DerivLosser, prediction, truth []float64) (derivative, fdDerivative []float64) {
	if len(prediction) != len(truth) {
		panic("prediction and truth are not the same length")
	}
	derivative = make([]float64, len(prediction))
	losser.LossDeriv(prediction, truth, derivative)

	fdDerivative = make([]float64, len(prediction))
	newDerivative1 := make([]float64, len(prediction))
	newDerivative2 := make([]float64, len(prediction))
	for i := range prediction {
		prediction[i] += FDStep
		loss1 := losser.LossDeriv(prediction, truth, newDerivative1)
		prediction[i] -= 2 * FDStep
		loss2 := losser.LossDeriv(prediction, truth, newDerivative2)
		prediction[i] += FDStep
		fdDerivative[i] = (loss1 - loss2) / (2 * FDStep)
	}
	return
}

func TestSquaredDistance(t *testing.T) {
	prediction := []float64{1, 2, 3}
	truth := []float64{1.1, 2.2, 2.7}
	trueloss := (.1*.1 + .2*.2 + .3*.3) / 3
	derivative := []float64{0, 0, 0}

	sq := SquaredDistance{}
	loss := sq.Loss(prediction, truth)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from Loss()")
	}

	loss = sq.LossDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from LossDeriv()")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. deriv: %v, fdDeriv: %v ", derivative, fdDerivative)
	}

	err := common.InterfaceTestMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}

	truth = []float64{1, 2, 3}
	loss = sq.LossDeriv(prediction, truth, derivative)
	if loss != 0 {
		t.Errorf("Non-zero loss for equal pred and truth")
	}
	for _, val := range derivative {
		if val != 0 {
			t.Errorf("Non-zero derivative for equal pred and truth")
		}
	}
}

func TestManhattanDistance(t *testing.T) {
	prediction := []float64{1, 2, 3}
	truth := []float64{1.1, 2.2, 2.7}
	trueloss := (.1 + .2 + .3) / 3
	derivative := []float64{0, 0, 0}

	sq := ManhattanDistance{}
	loss := sq.Loss(prediction, truth)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from Loss()")
	}

	loss = sq.LossDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from LossDeriv()")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}

	err := common.InterfaceTestMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}

	truth = []float64{1, 2, 3}
	loss = sq.LossDeriv(prediction, truth, derivative)
	if loss != 0 {
		t.Errorf("Non-zero loss for equal pred and truth")
	}
	for _, val := range derivative {
		if val != 0 {
			t.Errorf("Non-zero derivative for equal pred and truth")
		}
	}
}

func TestRelativeSquared(t *testing.T) {
	tol := 1e-2
	prediction := []float64{1, -2, 3}
	truth := []float64{1.1, -2.2, 2.7}
	trueloss := ((.1/(1.1+tol))*(.1/(1.1+tol)) + (.2/(2.2+tol))*(.2/(2.2+tol)) + (.3/(2.7+tol))*(.3/(2.7+tol))) / 3
	derivative := []float64{0, 0, 0}

	sq := RelativeSquared(tol)
	loss := sq.Loss(prediction, truth)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from Loss(). Expected %v, Found: %v", trueloss, loss)
	}

	loss = sq.LossDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from LossDeriv()")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}

	err := common.InterfaceTestMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}

func TestRelativeLog(t *testing.T) {
	tol := 1e-2
	prediction := []float64{1, -2, 3}
	truth := []float64{1.1, -2.2, 2.7}
	trueloss := ((.1/(1.1+tol))*(.1/(1.1+tol)) + (.2/(2.2+tol))*(.2/(2.2+tol)) + (.3/(2.7+tol))*(.3/(2.7+tol))) / 3
	trueloss = math.Log(trueloss + 1)
	derivative := []float64{0, 0, 0}

	sq := RelativeLog(tol)
	loss := sq.Loss(prediction, truth)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from Loss(). Expected %v, Found: %v", trueloss, loss)
	}

	loss = sq.LossDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from LossDeriv()")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}

	err := common.InterfaceTestMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling: " + err.Error())
	}
}

func TestLogSquared(t *testing.T) {
	prediction := []float64{1, -2, 3}
	truth := []float64{1.1, -2.2, 2.7}
	trueloss := (math.Log(.1*.1+1) + math.Log(.2*.2+1) + math.Log(.3*.3+1)) / 3
	derivative := []float64{0, 0, 0}

	sq := LogSquared{}
	loss := sq.Loss(prediction, truth)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from Loss(). Expected %v, Found: %v", trueloss, loss)
	}

	loss = sq.LossDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("loss doesn't match from LossDeriv()")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}
	err := common.InterfaceTestMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}
