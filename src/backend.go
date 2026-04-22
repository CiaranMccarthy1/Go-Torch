package gotorch

import (
	"runtime"
	"sync"
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

	numCPU := runtime.GOMAXPROCS(0)
	chunkSize := (m + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	for w := 0; w < numCPU; w++ {
		startRow := w * chunkSize
		endRow := startRow + chunkSize
		if startRow >= m {
			break
		}
		if endRow > m {
			endRow = m
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				rowOffsetA := i * k
				for j := 0; j < n; j++ {
					rowOffsetB := j * k
					var sum float32

					x := 0
					for ; x <= k-4; x += 4 {
						sum += a[rowOffsetA+x] * bT[rowOffsetB+x]
						sum += a[rowOffsetA+x+1] * bT[rowOffsetB+x+1]
						sum += a[rowOffsetA+x+2] * bT[rowOffsetB+x+2]
						sum += a[rowOffsetA+x+3] * bT[rowOffsetB+x+3]
					}
					for ; x < k; x++ {
						sum += a[rowOffsetA+x] * bT[rowOffsetB+x]
					}
					out[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return out
}

func (b *CPUBackend) MatMulBackward(a, bData, gradOut []float32, m, k, n int, needGradA, needGradB bool) ([]float32, []float32) {
	var gradA []float32
	var gradB []float32

	if needGradA {
		gradA = make([]float32, m*k)
		for i := 0; i < m; i++ {
			for x := 0; x < k; x++ {
				var sum float32
				for j := 0; j < n; j++ {
					sum += gradOut[i*n+j] * bData[x*n+j]
				}
				gradA[i*k+x] = sum
			}
		}
	}

	if needGradB {
		gradB = make([]float32, k*n)
		for x := 0; x < k; x++ {
			for j := 0; j < n; j++ {
				var sum float32
				for i := 0; i < m; i++ {
					sum += a[i*k+x] * gradOut[i*n+j]
				}
				gradB[x*n+j] = sum
			}
		}
	}

	return gradA, gradB
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
