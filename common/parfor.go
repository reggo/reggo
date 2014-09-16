package common

import (
	"runtime"
	"sync"
	"sync/atomic"
)

/*

func addSliceParallel(slice float64) float64 {
	sum := 0.0
	c := make(chan float64)
	wg := &sync.WaitGroup{}
	f := func(start, end int) {
		wg.Add()
		addSliceInds(slice, start, end, c)
	}
	for s := range c {
		sum += s
	}
	return sum
}

func addSliceInds(slice []float64, start, end int, c <-chan float64) {
	sum := 0.0
	for i := start; i < end; i++ {
		sum += slice[i]
	}
	c <- sum
}
*/

// GetGrainSize returns a reasonable value to use in Grain
func GetGrainSize(nSamples, minGrainSize, maxGrainSize int) int {
	procs := runtime.GOMAXPROCS(0)
	grainPerProc := nSamples / procs
	if grainPerProc < minGrainSize {
		return minGrainSize
	}
	if grainPerProc > maxGrainSize {
		return maxGrainSize
	}
	return grainPerProc
}

//TODO: May need to rethink grainsize thing

// ParallelFor computes the function f in parallel using chucks of the given size
func ParallelFor(n, grain int, f func(start, end int)) {
	P := runtime.GOMAXPROCS(0)
	idx := uint64(0)
	var wg sync.WaitGroup
	wg.Add(P)
	for p := 0; p < P; p++ {
		go func() {
			for {
				start := int(atomic.AddUint64(&idx, uint64(grain))) - grain
				if start >= n {
					break
				}
				end := start + grain
				if end > n {
					end = n
				}
				f(start, end)
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
