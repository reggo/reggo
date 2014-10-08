package train

import "math/rand"

// TODO: All this can be way simpler

type Sampler interface {
	Init(nSamples int) error
	// Gives a list of the indices to use. May assume that slice of ints will
	// not be modified.
	Iterate() []int
}

// Stochastic computes the function and gradient through evaluation of subsets
// of the data.
type Stochastic struct {
	BatchSize int
	// If false, random samples will be taken every time. If true, will iterate
	// through the data
	NoReplacement bool

	nSamples   int
	currSample int
	batch      []int
	perm       []int
}

func (s *Stochastic) Init(nSamples int) error {
	if s.BatchSize == 0 {
		s.BatchSize = 1
	}
	s.nSamples = nSamples
	s.batch = make([]int, s.BatchSize)
	if s.NoReplacement {
		return nil
	}
	s.perm = rand.Perm(nSamples)
	s.currSample = 0
	return nil
}

func (s *Stochastic) Iterate() []int {
	if s.NoReplacement {
		for i := range s.batch {
			s.batch[i] = rand.Intn(s.nSamples)
		}
		return s.batch
	}

	if s.BatchSize < len(s.perm) {
		n := copy(s.batch, s.perm)
		if n != s.BatchSize {
			panic("wrong number of elements copied")
		}
		s.perm = s.perm[s.BatchSize:]
		return s.batch
	}
	// Have to restart the permutation
	n := copy(s.batch, s.perm)
	s.perm = rand.Perm(s.nSamples)
	m := copy(s.batch[n:], s.perm[:s.BatchSize-n])
	if n+m != s.BatchSize {
		panic("wrong number of elements copied")
	}
	s.perm = s.perm[:s.BatchSize-n]
	return s.batch
}
