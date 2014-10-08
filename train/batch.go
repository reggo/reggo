package train

// Returns all of the elements at each iteration
type Batch struct {
	nSamples int
	batch    []int
}

func (b *Batch) Init(nSamples int) error {
	b.batch = make([]int, nSamples)
	for i := range b.batch {
		b.batch[i] = i
	}
	return nil
}

func (b *Batch) Iterate() []int {
	return b.batch
}
