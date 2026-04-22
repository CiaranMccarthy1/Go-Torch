package gotorch

import (
	"runtime"
	"sync"
)

const (
	matMulBlockK = 64
	matMulBlockN = 64
)

type TensorStorage interface {
	Data() []float32
}

type Backend interface {
	Name() string
	NewStorage(data []float32, size int) TensorStorage
	Transpose(data []float32, rows, cols int) []float32
	MatMulForward(a, b []float32, m, k, n int) []float32
	MatMulBackward(a, b, gradOut []float32, m, k, n int, needGradA, needGradB bool) ([]float32, []float32)
}

type sliceStorage struct {
	data []float32
}

func (s *sliceStorage) Data() []float32 {
	return s.data
}

type CPUBackend struct{}

func (b *CPUBackend) Name() string {
	return "cpu"
}

func (b *CPUBackend) NewStorage(data []float32, size int) TensorStorage {
	if data == nil {
		data = make([]float32, size)
	}
	return &sliceStorage{data: data}
}

func (b *CPUBackend) Transpose(data []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return out
}

func (b *CPUBackend) MatMulForward(a, bData []float32, m, k, n int) []float32 {
	bT := b.Transpose(bData, k, n)
	out := make([]float32, m*n)

	parallelFor(m, func(start, end int) {
		for i := start; i < end; i++ {
			aRow := a[i*k : (i+1)*k]
			outRow := out[i*n : (i+1)*n]

			for jBase := 0; jBase < n; jBase += matMulBlockN {
				jEnd := minInt(jBase+matMulBlockN, n)
				for kBase := 0; kBase < k; kBase += matMulBlockK {
					kEnd := minInt(kBase+matMulBlockK, k)
					for j := jBase; j < jEnd; j++ {
						bRow := bT[j*k : (j+1)*k]
						sum := outRow[j]

						x := kBase
						for ; x <= kEnd-4; x += 4 {
							sum += aRow[x] * bRow[x]
							sum += aRow[x+1] * bRow[x+1]
							sum += aRow[x+2] * bRow[x+2]
							sum += aRow[x+3] * bRow[x+3]
						}
						for ; x < kEnd; x++ {
							sum += aRow[x] * bRow[x]
						}

						outRow[j] = sum
					}
				}
			}
		}
	})

	return out
}

func (b *CPUBackend) MatMulBackward(a, bData, gradOut []float32, m, k, n int, needGradA, needGradB bool) ([]float32, []float32) {
	var gradA []float32
	var gradB []float32

	if needGradA {
		gradA = make([]float32, m*k)

		parallelFor(m, func(start, end int) {
			for i := start; i < end; i++ {
				gradARow := gradA[i*k : (i+1)*k]
				gradOutRow := gradOut[i*n : (i+1)*n]

				for kBase := 0; kBase < k; kBase += matMulBlockK {
					kEnd := minInt(kBase+matMulBlockK, k)
					for jBase := 0; jBase < n; jBase += matMulBlockN {
						jEnd := minInt(jBase+matMulBlockN, n)
						for x := kBase; x < kEnd; x++ {
							bRow := bData[x*n : (x+1)*n]
							sum := gradARow[x]

							j := jBase
							for ; j <= jEnd-4; j += 4 {
								sum += gradOutRow[j] * bRow[j]
								sum += gradOutRow[j+1] * bRow[j+1]
								sum += gradOutRow[j+2] * bRow[j+2]
								sum += gradOutRow[j+3] * bRow[j+3]
							}
							for ; j < jEnd; j++ {
								sum += gradOutRow[j] * bRow[j]
							}

							gradARow[x] = sum
						}
					}
				}
			}
		})
	}

	if needGradB {
		gradB = make([]float32, k*n)

		aT := b.Transpose(a, m, k)
		gradOutT := b.Transpose(gradOut, m, n)

		parallelFor(k, func(start, end int) {
			for x := start; x < end; x++ {
				aCol := aT[x*m : (x+1)*m]
				gradBRow := gradB[x*n : (x+1)*n]

				for jBase := 0; jBase < n; jBase += matMulBlockN {
					jEnd := minInt(jBase+matMulBlockN, n)
					for j := jBase; j < jEnd; j++ {
						gradOutCol := gradOutT[j*m : (j+1)*m]
						var sum float32

						i := 0
						for ; i <= m-4; i += 4 {
							sum += aCol[i] * gradOutCol[i]
							sum += aCol[i+1] * gradOutCol[i+1]
							sum += aCol[i+2] * gradOutCol[i+2]
							sum += aCol[i+3] * gradOutCol[i+3]
						}
						for ; i < m; i++ {
							sum += aCol[i] * gradOutCol[i]
						}

						gradBRow[j] = sum
					}
				}
			}
		})
	}

	return gradA, gradB
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func parallelFor(total int, fn func(start, end int)) {
	if total <= 0 {
		return
	}

	if total < 32 {
		fn(0, total)
		return
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > total {
		workers = total
	}

	chunkSize := (total + workers - 1) / workers

	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunkSize
		if start >= total {
			break
		}
		end := minInt(start+chunkSize, total)

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}

	wg.Wait()
}

var defaultBackend Backend = &CPUBackend{}

func DefaultBackend() Backend {
	return defaultBackend
}

func SetDefaultBackend(backend Backend) {
	if backend == nil {
		panic("backend cannot be nil")
	}
	defaultBackend = backend
}

func resolveBackend(tensors ...*Tensor) Backend {
	for _, t := range tensors {
		if t != nil && t.Backend != nil {
			return t.Backend
		}
	}
	return DefaultBackend()
}
