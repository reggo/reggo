// package kernel implements a number of different kernel functions

package kernel

import (
	"encoding/gob"
	"math"

	"github.com/reggo/reggo/common"
)

func init() {
	gob.Register(SqExpIso{})
	common.Register(SqExpIso{})
}

// sqDist computes the squared distance between x-y. Assumes lengths are equal
func sqDist(x, y []float64) float64 {
	scale := 0.0
	sumSquares := 1.0
	for i, xi := range x {
		val := xi - y[i]
		if val == 0 {
			continue
		}
		absxi := math.Abs(val)
		if scale < absxi {
			sumSquares = 1 + sumSquares*(scale/absxi)*(scale/absxi)
			scale = absxi
		} else {
			sumSquares = sumSquares + (absxi/scale)*(absxi/scale)
		}
	}
	return scale * math.Sqrt(sumSquares)
}

// Kerneler is a type that can compute a kernel function between two
// locations
type Kerneler interface {
	Kernel(x, y []float64) float64
}

// A DistKerneler is a type that compute the kernel based on the distance
// between two points
type DistKerneler interface {
	KernelDist(dist float64) float64
}

// SqExpIso represents an isotropic squared exponential kernel
// Logs are used for improved numerical conditioning
type SqExpIso struct {
	LogVariance float64 // Log of the variance of the kernel
	LogLength   float64 // Log of the length scale of the kernel function
}

func (kernel SqExpIso) KernelDist(dist float64) float64 {
	return math.Exp(kernel.LogKernelDist(dist))
}

// LogKernelDist returns the log of the distance
func (kernel SqExpIso) LogKernelDist(dist float64) float64 {

	logDist := math.Log(dist)
	distOverVariance := math.Exp(logDist - kernel.LogVariance)
	return -0.5*distOverVariance*distOverVariance + 2*kernel.LogLength
}

func (kernel SqExpIso) LogKernel(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("kernel: length mismatch")
	}
	dist := sqDist(x, y)
	return kernel.LogKernelDist(dist)
}

func (kernel SqExpIso) Kernel(x, y []float64) float64 {
	return math.Exp(kernel.LogKernel(x, y))
}
