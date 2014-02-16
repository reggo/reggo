package regularize

import (
	"math"

	"github.com/gonum/floats"
)

// Regularizer is a type that puts pressure on the values of
// parameters to prevent overfitting
type Regularizer interface {
	// How much loss is generated from the value of the parameters
	Loss(parameters []float64) float64

	// Returns the value of the loss and puts dLossDParameters
	// in place into the second argument. Writer may assume that
	// len(parameters) == len(derivative), but should not assume
	// that derivative is all zeros
	LossDeriv(parameters, derivative []float64) float64

	// LossAddDeriv adds the derivative rather than storing in place
	// TODO: Should this be here?
	LossAddDeriv(parameters, derivative []float64) float64
}

// TwoNorm gives the result of  ɣ||w||_2^2
type TwoNorm struct {
	Gamma float64 // Relative weight compared to loss function
}

func (t TwoNorm) Loss(parameters []float64) float64 {
	return t.Gamma * math.Pow(floats.Norm(parameters, 2), 2)
}

func (t TwoNorm) LossDeriv(parameters, derivative []float64) float64 {
	loss := t.Loss(parameters)
	for i, p := range parameters {
		derivative[i] = t.Gamma * 2 * p
	}
	return loss
}

func (t TwoNorm) LossAddDeriv(parameters, derivative []float64) float64 {
	loss := t.Loss(parameters)
	for i, p := range parameters {
		derivative[i] += t.Gamma * 2 * p
	}
	return loss
}

// TwoNorm gives the result of  ɣ||w||_2^2
type OneNorm struct {
	Gamma float64 // Relative weight compared to loss function
}

func (o OneNorm) Loss(parameters []float64) float64 {
	return o.Gamma * floats.Norm(parameters, 1)
}

func (o OneNorm) LossDeriv(parameters, derivative []float64) float64 {
	loss := o.Gamma * floats.Norm(parameters, 2)
	for i, p := range parameters {
		derivative[i] = o.Gamma * p
	}
	return loss
}

func (o OneNorm) LossAddDeriv(parameters, derivative []float64) float64 {
	loss := o.Gamma * floats.Norm(parameters, 2)
	for i, p := range parameters {
		derivative[i] += o.Gamma * p
	}
	return loss
}

// None represents no regularizer
type None struct{}

func (n None) Loss(parameters []float64) float64 {
	return 0
}

func (n None) LossDeriv(parameters, derivative []float64) float64 {
	for i := range derivative {
		derivative[i] = 0
	}
	return 0
}

func (n None) LossAddDeriv(parameters, derivative []float64) float64 {
	// Don't need to modify derivative at all
	return 0
}
