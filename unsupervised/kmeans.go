package kmeans

import (
	"math/rand"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

type KMeans struct {
	Clusters int

	centroids *mat64.Dense
}

func (k *KMeans) Train(inputs *common.RowMatrix) {
	nSamples, inputDim := inputs.Dims()
	centroids := make([][]float64, k.Clusters)

	// Assign the centroids to random data point to start
	perm := rand.Perm(nSamples)
	for i := 0; i < k.Clusters; i++ {
		data := make([]float64, inputDim)
		idx := perm[i]
		inputs.Row(data, idx)
		centroids[i] = data
	}

	distances := mat64.NewDense(nSamples, k.Clusters, nil)
	row := make([]float64, inputDim)
	for i := 0; i < nSamples; i++ {
		inputs.Row(row, i)
		for j := 0; j < k.Clusters; j++ {
			d := floats.Distance(row, centroids[j])
			distances.Set(i, j, d)
		}
	}

}
