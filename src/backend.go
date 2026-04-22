package gotorch

import (
	"math"
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
	ZeroStorage(size int) TensorStorage
	CopyToHost(s TensorStorage) []float32
	CopyToDevice(data []float32) TensorStorage
	Transpose(a TensorStorage, rows, cols int) TensorStorage
	MatMulForward(a, b TensorStorage, m, k, n int) TensorStorage
	MatMulBackward(a, b, gradOut TensorStorage, m, k, n int, needGradA, needGradB bool) (TensorStorage, TensorStorage)
	ReLUForward(a TensorStorage) TensorStorage
	ReLUBackward(input, gradOut TensorStorage) TensorStorage
	EmbedForward(weights, indices TensorStorage, batch, dim int) TensorStorage
	EmbedBackward(weights, indices, gradOut TensorStorage, batch, dim int) TensorStorage
	AddInPlace(dst TensorStorage, src TensorStorage)
	ZeroBuffer(s TensorStorage)
	AdamStep(param, grad, m, v TensorStorage, lr, beta1, beta2, eps float32, step int)
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

func (b *CPUBackend) ZeroStorage(size int) TensorStorage {
	return &sliceStorage{data: make([]float32, size)}
}

func (b *CPUBackend) CopyToHost(s TensorStorage) []float32 {
	if s == nil {
		return nil
	}
	return s.Data()
}

func (b *CPUBackend) CopyToDevice(data []float32) TensorStorage {
	if data == nil {
		return &sliceStorage{data: nil}
	}
	return &sliceStorage{data: data}
}

func (b *CPUBackend) Transpose(a TensorStorage, rows, cols int) TensorStorage {
	data := a.Data()
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return &sliceStorage{data: out}
}

func (cpu *CPUBackend) MatMulForward(aStorage, bStorage TensorStorage, m, k, n int) TensorStorage {
	aData := aStorage.Data()
	bT := cpu.Transpose(bStorage, k, n).Data()
	out := make([]float32, m*n)

	parallelFor(m, func(start, end int) {
		for i := start; i < end; i++ {
			aRow := aData[i*k : (i+1)*k]
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

	return &sliceStorage{data: out}
}

func (cpu *CPUBackend) MatMulBackward(aStorage, bStorage, gradOut TensorStorage, m, k, n int, needGradA, needGradB bool) (TensorStorage, TensorStorage) {
	bData := bStorage.Data()
	gradOutData := gradOut.Data()

	var gradA TensorStorage
	var gradB TensorStorage

	if needGradA {
		gradAData := make([]float32, m*k)

		parallelFor(m, func(start, end int) {
			for i := start; i < end; i++ {
				gradARow := gradAData[i*k : (i+1)*k]
				gradOutRow := gradOutData[i*n : (i+1)*n]

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

		gradA = &sliceStorage{data: gradAData}
	}

	if needGradB {
		gradBData := make([]float32, k*n)

		aT := cpu.Transpose(aStorage, m, k).Data()
		gradOutT := cpu.Transpose(gradOut, m, n).Data()

		parallelFor(k, func(start, end int) {
			for x := start; x < end; x++ {
				aCol := aT[x*m : (x+1)*m]
				gradBRow := gradBData[x*n : (x+1)*n]

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

		gradB = &sliceStorage{data: gradBData}
	}

	return gradA, gradB
}

func (b *CPUBackend) ReLUForward(a TensorStorage) TensorStorage {
	aData := a.Data()
	out := make([]float32, len(aData))
	for i, v := range aData {
		if v > 0 {
			out[i] = v
		}
	}
	return &sliceStorage{data: out}
}

func (b *CPUBackend) ReLUBackward(input, gradOut TensorStorage) TensorStorage {
	inputData := input.Data()
	gradOutData := gradOut.Data()
	out := make([]float32, len(inputData))
	for i, v := range inputData {
		if v > 0 {
			out[i] = gradOutData[i]
		}
	}
	return &sliceStorage{data: out}
}

func (b *CPUBackend) EmbedForward(weights, indices TensorStorage, batch, dim int) TensorStorage {
	weightsData := weights.Data()
	indicesData := indices.Data()

	if len(indicesData) < batch {
		panic("indices storage shorter than batch size")
	}

	vocabSize := len(weightsData) / dim
	out := make([]float32, batch*dim)
	for i := 0; i < batch; i++ {
		idx := int(indicesData[i])
		if idx < 0 || idx >= vocabSize {
			panic("embedding index out of range")
		}
		copy(out[i*dim:(i+1)*dim], weightsData[idx*dim:(idx+1)*dim])
	}

	return &sliceStorage{data: out}
}

func (b *CPUBackend) EmbedBackward(weights, indices, gradOut TensorStorage, batch, dim int) TensorStorage {
	weightsData := weights.Data()
	indicesData := indices.Data()
	gradOutData := gradOut.Data()

	if len(indicesData) < batch {
		panic("indices storage shorter than batch size")
	}

	vocabSize := len(weightsData) / dim
	gradWeights := make([]float32, len(weightsData))
	for i := 0; i < batch; i++ {
		idx := int(indicesData[i])
		if idx < 0 || idx >= vocabSize {
			panic("embedding index out of range")
		}
		for k := 0; k < dim; k++ {
			gradWeights[idx*dim+k] += gradOutData[i*dim+k]
		}
	}

	return &sliceStorage{data: gradWeights}
}

func (b *CPUBackend) AddInPlace(dst TensorStorage, src TensorStorage) {
	if dst == nil || src == nil {
		return
	}

	dstData := dst.Data()
	srcData := src.Data()
	if len(dstData) != len(srcData) {
		panic("storage size mismatch in AddInPlace")
	}

	for i, v := range srcData {
		dstData[i] += v
	}
}

func (b *CPUBackend) ZeroBuffer(s TensorStorage) {
	if s == nil {
		return
	}
	for i := range s.Data() {
		s.Data()[i] = 0
	}
}

func (b *CPUBackend) AdamStep(param, grad, m, v TensorStorage, lr, beta1, beta2, eps float32, step int) {
	if param == nil || grad == nil || m == nil || v == nil {
		return
	}

	paramData := param.Data()
	gradData := grad.Data()
	mData := m.Data()
	vData := v.Data()

	if len(paramData) != len(gradData) || len(paramData) != len(mData) || len(paramData) != len(vData) {
		panic("storage size mismatch in AdamStep")
	}

	beta1Pow := float32(math.Pow(float64(beta1), float64(step)))
	beta2Pow := float32(math.Pow(float64(beta2), float64(step)))
	oneMinusBeta1Pow := float32(1.0) - beta1Pow
	oneMinusBeta2Pow := float32(1.0) - beta2Pow

	for i := range paramData {
		g := gradData[i]
		mData[i] = beta1*mData[i] + (1-beta1)*g
		vData[i] = beta2*vData[i] + (1-beta2)*g*g

		mHat := mData[i] / oneMinusBeta1Pow
		vHat := vData[i] / oneMinusBeta2Pow

		paramData[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
	}
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
