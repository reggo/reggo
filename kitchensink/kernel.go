package kitchensink

import (
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// A kitchen sink kernel is a type that can generate random feature vectors
// to train with the kitchen sink algorithm
type Kernel interface {
	// Generate a list of n features with dimension dim. Implementer
	// may assume that features is a n x dim matrix
	Generate(n, dim int, features *mat64.Dense)
}

// IsoSqExp is an isotropic squared exponential kernel, where
// k(x,x') = exp( -||x - x'||^2 / (2 σ^2))
type IsoSqExp struct {
	LogScale float64 // The log of the bandwidth (log(σ))
	//scale    float64
}

// Generate generates a list of n random features given an input dimension d
func (iso IsoSqExp) Generate(n int, dim int, features *mat64.Dense) {
	scale := math.Exp(iso.LogScale)

	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			features.Set(i, j, rand.NormFloat64()*scale)
		}
	}
}
