package loss

import (
	"encoding/gob"
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

// init registers all of the types into the common registry for
// encoding and decoding
func init() {
	gob.Register(SquaredDistance{})
	gob.Register(ManhattanDistance{})
	gob.Register(RelativeSquared(0))
	gob.Register(LogSquared{})
	common.Register(SquaredDistance{})
	common.Register(ManhattanDistance{})
	a := RelativeSquared(0)
	common.Register(a)
	b := RelativeLog(0)
	common.Register(b)
	common.Register(LogSquared{})
}

var lenMismatch string = "length mismatch"

// Losser is an interface for a loss function.
// A loss function is a measure of the quality of a prediction, with
// a lower value of loss being better. Typically, the loss is zero
// iff prediction == truth, and is always non-negative
// A Losser will panic if len(prediction) != len(truth). The losser
// should not modify the slice values
type Losser interface {
	Loss(prediction, truth []float64) float64
}

// A DerivLosser is a loss function which can the loss and also the derivative
// of the loss function with respect to the prediction. The derivative
// is put in place into the derivative slice.
// The DerivLosser will panic if len(prediction), len(truth), and
// len(derivative) are not all equal
type DerivLosser interface {
	Losser
	LossDeriv(prediction, truth, derivative []float64) float64
}

// A ConvexDerivLosser is a loss function that is convex in the prediction
type ConvexDerivLosser interface {
	DerivLosser
	Convex()
}

// TODO: This should be divided by the square root of the length

// SquaredDistance is the same as the two-norm of (pred - truth) divided by the
// length
type SquaredDistance struct{}

func (SquaredDistance) Loss(prediction, truth []float64) (loss float64) {
	if len(prediction) != len(truth) {
		panic(lenMismatch)
	}
	for i := range prediction {
		diff := prediction[i] - truth[i]
		loss += diff * diff
	}
	loss /= float64(len(prediction))
	return loss
}

func (SquaredDistance) LossDeriv(prediction, truth, derivative []float64) (loss float64) {
	if len(prediction) != len(truth) || len(prediction) != len(derivative) {
		panic(lenMismatch)
	}
	for i := range prediction {
		diff := prediction[i] - truth[i]
		derivative[i] = diff
		loss += diff * diff
	}
	loss /= float64(len(prediction))
	for i := range derivative {
		derivative[i] /= float64(len(prediction)) / 2
	}
	return loss
}

// TODO: Make this a mutable symmetric

func (SquaredDistance) LossDerivHess(prediction, truth, derivative []float64, hessian *mat64.Dense) (loss float64) {
	if len(prediction) != len(truth) || len(prediction) != len(derivative) {
		panic(lenMismatch)
	}
	n, m := hessian.Dims()
	if len(prediction) != n {
		panic(lenMismatch)
	}
	if len(prediction) != m {
		panic(lenMismatch)
	}
	for i := range prediction {
		diff := prediction[i] - truth[i]
		derivative[i] = diff
		loss += diff * diff
	}

	nFloat := float64(n)
	loss /= nFloat

	corr := 2 / nFloat

	for i := range derivative {
		derivative[i] *= corr
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				hessian.Set(i, j, corr)
			} else {
				hessian.Set(i, j, 0)
			}
		}
	}

	return loss
}

// Convex allows ManhattanDistance to be a ConvexDerivLosser
func (SquaredDistance) Convex() {}

// Manhattan distance is the same as the one-norm of (pred - truth)
type ManhattanDistance struct{}

func (ManhattanDistance) Loss(prediction, truth []float64) float64 {
	if len(prediction) != len(truth) {
		panic(lenMismatch)
	}
	var loss float64
	for i, val := range prediction {
		loss += math.Abs(val - truth[i])
	}
	loss /= float64(len(prediction))
	return loss
}

func (ManhattanDistance) LossDeriv(prediction, truth, derivative []float64) (loss float64) {
	if len(prediction) != len(truth) || len(prediction) != len(derivative) {
		panic(lenMismatch)
	}
	for i := range prediction {
		loss += math.Abs(prediction[i] - truth[i])
		if prediction[i] > truth[i] {
			derivative[i] = 1.0 / float64(len(prediction))
		} else if prediction[i] < truth[i] {
			derivative[i] = -1.0 / float64(len(prediction))
		} else {
			derivative[i] = 0
		}
	}
	loss /= float64(len(prediction))
	return loss
}

// Convex allows ManhattanDistance to be a ConvexDerivLosser
func (ManhattanDistance) Convex() {}

// Relative squared is the relative error with the value of RelativeSquared added in the denominator
type RelativeSquared float64

func (r RelativeSquared) Loss(prediction, truth []float64) float64 {
	if len(prediction) != len(truth) {
		panic(lenMismatch)
	}
	var loss float64
	for i, pred := range prediction {
		tr := truth[i]
		denom := math.Abs(tr) + float64(r)
		diff := pred - tr
		diffOverDenom := diff / denom
		loss += diffOverDenom * diffOverDenom
	}
	loss /= float64(len(prediction))
	return loss
}

func (r RelativeSquared) LossDeriv(prediction, truth, derivative []float64) (loss float64) {
	if len(prediction) != len(truth) || len(prediction) != len(derivative) {
		panic(lenMismatch)
	}
	nSamples := float64(len(prediction))
	for i := range prediction {
		denom := math.Abs(truth[i]) + float64(r)
		diff := prediction[i] - truth[i]

		diffOverDenom := diff / denom

		loss += diffOverDenom * diffOverDenom
		derivative[i] = 2 * diffOverDenom / denom / nSamples
	}
	loss /= nSamples
	return loss
}

// LogRelative finds the relative difference between the two samples and takes the log
// of the absolute value of the difference as the loss function
type RelativeLog float64

func (l RelativeLog) Loss(prediction, truth []float64) float64 {
	var loss float64
	for i, pred := range prediction {
		tr := truth[i]
		denom := math.Abs(tr) + float64(l)
		diff := pred - tr
		diffOverDenom := diff / denom
		loss += diffOverDenom * diffOverDenom
	}
	loss /= float64(len(prediction))
	loss = math.Log(loss + 1)
	return loss
}

func (l RelativeLog) LossDeriv(prediction, truth, derivative []float64) (loss float64) {
	nDim := float64(len(prediction))
	for i := range prediction {
		denom := math.Abs(truth[i]) + float64(l)
		diff := prediction[i] - truth[i]

		diffOverDenom := diff / denom
		loss += diffOverDenom * diffOverDenom
		derivative[i] = (2 / nDim) * diffOverDenom / denom
	}
	loss /= nDim
	// d (log Loss) / dx = 1/loss * dLoss/dx
	for i := range prediction {
		derivative[i] /= (loss + 1)
	}
	loss = math.Log(loss + 1) // so the minimum loss is zero
	return loss
}

// LogSquared uses log(1 + diff*diff) so that really high losses aren't as important
type LogSquared struct{}

func (LogSquared) Loss(prediction, truth []float64) float64 {
	var loss float64
	for i, pred := range prediction {
		diff := pred - truth[i]
		diffSqPlus1 := diff*diff + 1
		loss += math.Log(diffSqPlus1)
	}
	loss /= float64(len(prediction))
	return loss
}

func (LogSquared) LossDeriv(prediction, truth, deravitive []float64) (loss float64) {
	nSamples := float64(len(prediction))
	for i := range prediction {
		diff := prediction[i] - truth[i]
		diffSqPlus1 := diff*diff + 1
		loss += math.Log(diffSqPlus1)
		deravitive[i] = 2 / diffSqPlus1 * diff / nSamples
	}
	loss /= nSamples
	return loss
}
