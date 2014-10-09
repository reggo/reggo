package train

import (
	"errors"
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
)

// Optimizes the gradient. Note the if Stochastic sampler is used,
// the function is stochastic and may fail with normal optimization
// algorithms.
type GradOptimizable struct {
	// Must be set
	Trainable Trainable
	Inputs    common.RowMatrix
	Outputs   common.RowMatrix
	Weights   []float64

	// Will be defaulted
	NumWorkers  int // defaults to off
	Sampler     Sampler
	Losser      loss.DerivLosser
	Regularizer regularize.Regularizer

	nSamples  int
	inputDim  int
	outputDim int
	features  *mat64.Dense

	sendWork       chan<- batchSend
	receiveWork    <-chan batchSend
	regularizeChan chan<- batchSend
	regDone        <-chan batchSend
	quit           chan<- struct{}

	batches []batchSend
	wg      *sync.WaitGroup

	grainSize int
}

func (g *GradOptimizable) Init() error {
	if g.Losser == nil {
		g.Losser = loss.SquaredDistance{}
	}
	if g.Regularizer == nil {
		g.Regularizer = regularize.None{}
	}
	if g.Sampler == nil {
		g.Sampler = &Batch{}
	}

	if g.Inputs == nil {
		return errors.New("No input data")
	}

	nSamples, _ := g.Inputs.Dims()
	if nSamples == 0 {
		return errors.New("No input data")
	}
	if g.NumWorkers == 0 {
		g.NumWorkers = 1
	}

	outputSamples, outputDim := g.Outputs.Dims()
	if outputSamples != nSamples {
		return errors.New("gradoptimize: input and output row mismatch")
	}

	nParameters := g.Trainable.NumParameters()

	batches := make([]batchSend, g.NumWorkers+1) // +1 is for regularizer
	for i := range batches {
		batches[i].deriv = make([]float64, nParameters)
	}
	g.batches = batches

	g.grainSize = g.Trainable.GrainSize()

	g.Sampler.Init(nSamples)

	g.features = FeaturizeTrainable(g.Trainable, g.Inputs, nil)

	work := make(chan batchSend, g.NumWorkers)
	done := make(chan batchSend, g.NumWorkers)
	regularizeChan := make(chan batchSend, 1)
	regDone := make(chan batchSend, 1)
	quit := make(chan struct{})

	g.sendWork = work
	g.receiveWork = done
	g.quit = quit
	g.regularizeChan = regularizeChan
	g.regDone = regDone

	// launch workers
	for worker := 0; worker < g.NumWorkers; worker++ {
		go func(outputDim, nParameterss int) {
			lossDeriver := g.Trainable.NewLossDeriver()
			predOutput := make([]float64, outputDim)
			dLossDPred := make([]float64, outputDim)
			dLossDParam := make([]float64, nParameters)
			outputs := make([]float64, outputDim)
			for {
				select {
				case w := <-work:
					// Zero out existing derivative
					w.loss = 0
					for i := range w.deriv {
						w.deriv[i] = 0
					}
					for _, idx := range w.idxs {
						lossDeriver.Predict(w.parameters, g.features.RowView(idx), predOutput)
						g.Outputs.Row(outputs, idx)
						loss := g.Losser.LossDeriv(predOutput, outputs, dLossDPred)
						if g.Weights == nil {
							w.loss += loss
						} else {
							w.loss += g.Weights[idx] * loss
						}
						lossDeriver.Deriv(w.parameters, g.features.RowView(idx), predOutput, dLossDPred, dLossDParam)
						if g.Weights != nil {
							floats.Scale(g.Weights[idx], dLossDParam)
						}
						floats.Add(w.deriv, dLossDParam)
					}
					// Send the result back
					done <- w
				case <-quit:
					return
				}
			}
		}(outputDim, nParameters)
	}

	// launch regularizer
	go func() {
		for {
			select {
			case w := <-regularizeChan:
				loss := g.Regularizer.LossDeriv(w.parameters, w.deriv)
				w.loss = loss
				regDone <- w
			case <-quit:
				return
			}
		}
	}()
	return nil
}

// Close stops all the workers. Should be called when done using
func (g *GradOptimizable) Close() {
	close(g.quit)
}

func (g *GradOptimizable) F(params []float64) float64 {
	deriv := make([]float64, len(params))
	return g.FDf(params, deriv)
}

func (g *GradOptimizable) FDf(params []float64, deriv []float64) float64 {
	inds := g.Sampler.Iterate()
	total := len(inds)

	var totalLoss float64
	for i := range deriv {
		deriv[i] = 0
	}

	// Send the regularizer
	g.batches[0].parameters = params
	g.regularizeChan <- g.batches[0]

	// Send initial batches out
	var initBatches int
	var lastSent int
	for i := 0; i < g.NumWorkers; i++ {
		if lastSent == total {
			break
		}
		add := g.grainSize
		if lastSent+add >= total {
			add = total - lastSent
		}
		initBatches++
		g.batches[i+1].idxs = inds[lastSent : lastSent+add]
		g.batches[i+1].parameters = params
		g.sendWork <- g.batches[i+1]
		lastSent += add
	}

	// Collect the batches and resend out
	for lastSent < total {
		batch := <-g.receiveWork
		totalLoss += batch.loss
		floats.Add(deriv, batch.deriv)
		add := g.grainSize
		if lastSent+add >= total {
			add = total - lastSent
		}
		batch.idxs = inds[lastSent : lastSent+add]
		g.sendWork <- batch
		lastSent += add
	}

	// All inds sent, so just weight for all the collection
	for i := 0; i < initBatches; i++ {
		batch := <-g.receiveWork
		totalLoss += batch.loss
		floats.Add(deriv, batch.deriv)
	}
	batch := <-g.regDone
	totalLoss += batch.loss
	floats.Add(deriv, batch.deriv)

	totalLoss /= float64(len(inds))
	floats.Scale(1/float64(len(inds)), deriv)
	return totalLoss
}

type batchSend struct {
	idxs       []int
	parameters []float64
	deriv      []float64
	loss       float64
}
