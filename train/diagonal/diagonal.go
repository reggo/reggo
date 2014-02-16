package diagonal

// This is here until gonum adds a diagonal

import (
	"github.com/gonum/matrix/mat64"
)

func NewDiagonal(n int, diag []float64) *Diagonal {
	if diag != nil && len(diag) != n {
		panic(mat64.ErrShape)
	}
	if diag == nil {
		diag = make([]float64, n)
	}
	return &Diagonal{n: n, diag: diag}
}

type Diagonal struct {
	diag []float64
	n    int
}

func (d *Diagonal) Dims() (int, int) {
	return d.n, d.n
}

func (d *Diagonal) At(i, j int) float64 {
	if i != j {
		return 0
	}
	return d.diag[i]
}

// Computes the inverse. Not an Inver because cannot store
// a general matrix
func (d *Diagonal) Inv(e *Diagonal) error {
	var err error
	for i, val := range e.diag {
		if val == 0 {
			err = mat64.ErrSingular
		}
		d.diag[i] = 1 / val
	}
	return err
}
