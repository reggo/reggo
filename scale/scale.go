package scale

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/reggo/reggo/common"

	"github.com/gonum/matrix/mat64"
)

// TODO: Should have a Scaler and a SettableScaler where SetScale returns a Scaler.
// FixedNormal (etc.). This should be fine, because most peolpe won't have to
// interface with those types.
// scale.Settable

func init() {
	gob.Register(None{})
	gob.Register(Linear{})
	gob.Register(Normal{})
	//gob.Register(&Probability{})

	common.Register(&None{})
	common.Register(&Linear{})
	common.Register(&Normal{})
	//common.Register(&Probability{})
}

// TODO: Check that scalers don't set IsScaled if there is an error, and add
// comment about the behavior
// TODO: Add comment that it is assumed if data can be scaled, it can also be unscaled without error

// IdenticalDimensions is an error type expressing that
// a dimension all had equal values. Dims is a list of unequal dimensions
type UniformDimension struct {
	Dims []int
}

func (i *UniformDimension) Error() string {
	return "Some dimensions had all values with the same entry"
}

type UnequalLength struct{}

func (u UnequalLength) Error() string {
	return "Data length mismatch"
}

// Scalar is an interface for transforming data so it is appropriately scaled
// for the machine learning algorithm. The data are a slice of data points.
// All of the data points must have equal lengths. An error is returned if
// some of the data have unequal lengths or if less than two data points are
// entered
type Scaler interface {
	Scale(point []float64) error      // Scales (in place) the data point
	Unscale(point []float64) error    // Unscales (in place) the data point
	IsScaled() bool                   // Returns true if the scale for this type has already been set
	Dimensions() int                  //Number of dimensions for wihich the data was scaled
	SetScale(data *mat64.Dense) error // Uses the input data to set the scale
}

type SliceError struct {
	Header string
	Idx    int
	Err    error
}

func (s *SliceError) Error() string {
	return fmt.Sprintf("%v: element %v, error %v", s.Header, s.Idx, s.Err)
}

type ErrorList []*SliceError

func (e ErrorList) Error() string {
	return fmt.Sprintf("%v errors found", len(e))
}

// ScaleData is a wrapper for scaling data in parallel.
// TODO: Make this work better so that if there is an error somewhere data isn't changed
func ScaleData(scaler Scaler, data *mat64.Dense) error {
	m := &sync.Mutex{}
	var e ErrorList
	f := func(start, end int) {
		for r := start; r < end; r++ {
			errTmp := scaler.Scale(data.RowView(r))
			if errTmp != nil {
				m.Lock()
				e = append(e, &SliceError{Header: "scale", Idx: r, Err: errTmp})
				m.Unlock()
			}
		}
	}

	nSamples, _ := data.Dims()
	grain := common.GetGrainSize(nSamples, 1, 500)
	common.ParallelFor(nSamples, grain, f)

	if len(e) != 0 {
		return e
	}
	return nil
}

// UnscaleData is a wrapper for unscaling data in parallel.
// TODO: Make this work better so that if there is an error somewhere data isn't changed
func UnscaleData(scaler Scaler, data *mat64.Dense) error {
	m := &sync.Mutex{}
	var e ErrorList
	f := func(start, end int) {
		for r := start; r < end; r++ {
			errTmp := scaler.Unscale(data.RowView(r))
			if errTmp != nil {
				m.Lock()
				e = append(e, &SliceError{Header: "scale", Idx: r, Err: errTmp})
				m.Unlock()
			}
		}
	}

	nSamples, _ := data.Dims()
	grain := common.GetGrainSize(nSamples, 1, 500)
	common.ParallelFor(nSamples, grain, f)
	if len(e) != 0 {
		return e
	}
	return nil
}

// ScaleTrainingData sets the scale of the scalers if they are not already set
// and then scales the data in inputs and outputs
// TODO: Change so that if any error occurs, scaling will be undone
// TODO: Make run concurrently
func ScaleTrainingData(inputs, outputs *mat64.Dense, inputScaler, outputScaler Scaler) error {
	var err error
	if !inputScaler.IsScaled() {
		err = inputScaler.SetScale(inputs)
		if err != nil {
			return err
		}
	}
	if !outputScaler.IsScaled() {
		err = outputScaler.SetScale(outputs)
		if err != nil {
			return err
		}
	}
	err = ScaleData(inputScaler, inputs)
	if err != nil {
		return err
	}
	err = ScaleData(outputScaler, outputs)
	if err != nil {
		UnscaleData(inputScaler, inputs)
		return err
	}
	return nil
}

func UnscaleTrainingData(inputs, outputs *mat64.Dense, inputScaler, outputScaler Scaler) error {
	UnscaleData(inputScaler, inputs)
	UnscaleData(outputScaler, outputs)
	return nil
}

// None is a type specifying no transformation of the input should be done
type None struct {
	Dim    int // Dimensions
	Scaled bool
}

func (n None) IsScaled() bool {
	return n.Scaled
}

func (n None) Scale(x []float64) error {
	return nil
}

func (n None) Unscale(x []float64) error {
	return nil
}

func (n None) Dimensions() int {
	return n.Dim
}

func (n *None) SetScale(data *mat64.Dense) error {
	rows, cols := data.Dims()
	if rows < 2 {
		return errors.New("scale: less than two inputs")
	}

	n.Dim = cols
	n.Scaled = true
	return nil
}

// Linear is a type for scaling the data to be between 0 and 1
type Linear struct {
	Min    []float64 // Maximum value of the data
	Max    []float64 // Minimum value of the data
	Scaled bool      // Flag if the scale has been set
	Dim    int       // Number of dimensions of the data
}

// IsScaled returns true if the scale has been set
func (l *Linear) IsScaled() bool {
	return l.Scaled
}

// Dimensions returns the length of the data point
func (l *Linear) Dimensions() int {
	return l.Dim
}

// SetScale sets a linear scale between 0 and 1. If no data
// points. If the minimum and maximum value are identical in
// a dimension, the minimum and maximum values will be set to
// that value +/- 0.5 and a
func (l *Linear) SetScale(data *mat64.Dense) error {

	rows, dim := data.Dims()
	if rows < 2 {
		return errors.New("scale: less than two inputs")
	}

	// Generate data for min and max if they don't already exist
	if len(l.Min) < dim {
		l.Min = make([]float64, dim)
	} else {
		l.Min = l.Min[0:dim]
	}
	if len(l.Max) < dim {
		l.Max = make([]float64, dim)
	} else {
		l.Max = l.Max[0:dim]
	}
	for i := range l.Min {
		l.Min[i] = math.Inf(1)
	}
	for i := range l.Max {
		l.Max[i] = math.Inf(-1)
	}
	// Find the minimum and maximum in each dimension
	for i := 0; i < rows; i++ {
		for j := 0; j < dim; j++ {
			val := data.At(i, j)
			if val < l.Min[j] {
				l.Min[j] = val
			}
			if val > l.Max[j] {
				l.Max[j] = val
			}
		}
	}
	l.Scaled = true
	l.Dim = dim

	var unifError *UniformDimension

	// Check that the maximum and minimum values are not identical
	for i := range l.Min {
		if l.Min[i] == l.Max[i] {
			if unifError == nil {
				unifError = &UniformDimension{}
			}
			unifError.Dims = append(unifError.Dims, i)
			l.Min[i] -= 0.5
			l.Max[i] += 0.5
		}
	}
	if unifError != nil {
		return unifError
	}
	return nil
}

// Scales the point returning an error if the length doesn't match
func (l *Linear) Scale(point []float64) error {
	if len(point) != l.Dim {
		return UnequalLength{}
	}
	for i, val := range point {
		point[i] = (val - l.Min[i]) / (l.Max[i] - l.Min[i])
	}
	return nil
}

func (l *Linear) Unscale(point []float64) error {
	if len(point) != l.Dim {
		return UnequalLength{}
	}
	for i, val := range point {
		point[i] = val*(l.Max[i]-l.Min[i]) + l.Min[i]
	}
	return nil
}

// Normal scales the data to have a mean of 0 and a variance of 1
// in each dimension
type Normal struct {
	Mu     []float64
	Sigma  []float64
	Dim    int
	Scaled bool
}

// IsScaled returns true if the scale has been set
func (n *Normal) IsScaled() bool {
	return n.Scaled
}

// Dimensions returns the length of the data point
func (n *Normal) Dimensions() int {
	return n.Dim
}

// SetScale Finds the appropriate scaling of the data such that the dataset has
// a mean of 0 and a variance of 1. If the standard deviation of any of
// the data is zero (all of the entries have the same value),
// the standard deviation is set to 1.0 and a UniformDimension error is
// returned
func (n *Normal) SetScale(data *mat64.Dense) error {

	rows, dim := data.Dims()
	if rows < 2 {
		return errors.New("scale: less than two inputs")
	}

	// Need to find the mean input and the std of the input
	mean := make([]float64, dim)
	for i := 0; i < rows; i++ {
		for j := 0; j < dim; j++ {
			mean[j] += data.At(i, j)
		}
	}
	for i := range mean {
		mean[i] /= float64(rows)
	}

	// TODO: Replace this with something that has better numerical properties
	std := make([]float64, dim)
	for i := 0; i < rows; i++ {
		for j := 0; j < dim; j++ {
			diff := data.At(i, j) - mean[j]
			std[j] += diff * diff
		}
	}
	for i := range std {
		std[i] /= float64(rows)
		std[i] = math.Sqrt(std[i])
	}
	n.Scaled = true
	n.Dim = dim

	var unifError *UniformDimension
	for i := range std {
		if std[i] == 0 {
			if unifError == nil {
				unifError = &UniformDimension{}
			}
			unifError.Dims = append(unifError.Dims, i)
			std[i] = 1.0
		}
	}

	n.Mu = mean
	n.Sigma = std
	if unifError != nil {
		return unifError
	}
	return nil
}

// Scale scales the data point
func (n *Normal) Scale(point []float64) error {
	if len(point) != n.Dim {
		return UnequalLength{}
	}
	for i := range point {
		point[i] = (point[i] - n.Mu[i]) / n.Sigma[i]
	}
	return nil
}

// Unscale unscales the data point
func (n *Normal) Unscale(point []float64) error {
	if len(point) != n.Dim {
		return UnequalLength{}
	}
	for i := range point {
		point[i] = point[i]*n.Sigma[i] + n.Mu[i]
	}
	return nil
}

/*
type ProbabilityDistribution interface {
	Fit([]float64) error
	CumProb(float64) float64
	Quantile(float64) float64
	Prob(float64) float64
}

// Probability scales the inputs based on the supplied
// probability distributions
type Probability struct {
	UnscaledDistribution []ProbabilityDistribution // Probabilitiy distribution from which the data come
	ScaledDistribution   []ProbabilityDistribution // Probability distribution to which the data should be scaled
	Dim                  int
	Scaled               bool
}

// IsScaled returns true if the scale has been set
func (p *Probability) IsScaled() bool {
	return p.Scaled
}

// Dimensions returns the length of the data point
func (p *Probability) Dimensions() int {
	return p.Dim
}

func (p *Probability) SetScale(data *mat64.Dense) error {
	err := checkInputs(data)
	if err != nil {
		return err
	}
	p.Dim = len(data[0])
	if len(p.UnscaledDistribution) != p.Dim {
		return errors.New("Number of unscaled probability distributions must equal dimension")
	}
	if len(p.ScaledDistribution) != p.Dim {
		return errors.New("Unscaled distribution not set")
	}

	tmp := make([]float64, len(data))
	for i := 0; i < p.Dim; i++ {
		// Collect all the data into tmp
		for j, point := range data {
			tmp[j] = point[i]
		}
		// Fit the probability distribution using the samples
		p.UnscaledDistribution[i].Fit(tmp)
	}
	return nil
}

func (p *Probability) Scale(point []float64) error {
	if len(point) != p.Dim {
		return UnequalLength{}
	}
	for i := range point {
		// Check that the point doesn't have zero probability
		if p.UnscaledDistribution[i].Prob(point[i]) == 0 {
			return errors.New("Zero probability point")
		}
		prob := p.UnscaledDistribution[i].CumProb(point[i])
		point[i] = p.ScaledDistribution[i].Quantile(prob)
		if math.IsInf(point[i], 0) {
			panic("inf point")
		}
		if math.IsNaN(point[i]) {
			panic("NaN point")
		}
	}
	return nil
}

func (p *Probability) Unscale(point []float64) error {
	if len(point) != p.Dim {
		return UnequalLength{}
	}
	for i := range point {
		// Check that the point doesn't have zero probability
		if p.UnscaledDistribution[i].Prob(point[i]) == 0 {
			return errors.New("Zero probability point")
		}
		prob := p.ScaledDistribution[i].CumProb(point[i])
		point[i] = p.UnscaledDistribution[i].Quantile(prob)
	}
	return nil
}
*/
