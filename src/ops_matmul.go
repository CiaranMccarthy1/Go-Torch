package gotorch

import (
	"fmt"
	"runtime"
	"sync"
)

type MatMulOp struct{}

func (op MatMulOp) Backward(t *Tensor) {
	a, b := t.Parents[0], t.Parents[1]
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]

	if a.ReqGrad {
		for i := 0; i < M; i++ {
			for k := 0; k < K; k++ {
				var sum float32
				for j := 0; j < N; j++ {
					sum += t.Grad[i*N+j] * b.Data[k*N+j]
				}
				AtomicAddFloat32(&a.Grad[i*K+k], sum)
			}
		}
	}
	if b.ReqGrad {
		for k := 0; k < K; k++ {
			for j := 0; j < N; j++ {
				var sum float32
				for i := 0; i < M; i++ {
					sum += a.Data[i*K+k] * t.Grad[i*N+j]
				}
				AtomicAddFloat32(&b.Grad[k*N+j], sum)
			}
		}
	}
}

func MatMul(a, b *Tensor) *Tensor {
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("Shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]

	// 1. Transpose B so we can access it sequentially (Row-Major)
	bT := Transpose(b)

	out := make([]float32, M*N)

	// 2. Setup Worker Pool
	numCPU := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup

	chunkSize := (M + numCPU - 1) / numCPU

	for w := 0; w < numCPU; w++ {
		startRow := w * chunkSize
		endRow := startRow + chunkSize
		if startRow >= M {
			break
		}
		if endRow > M {
			endRow = M
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				rowOffsetA := i * K
				for j := 0; j < N; j++ {
					rowOffsetB := j * K
					var sum float32

					k := 0
					for ; k <= K-4; k += 4 {
						sum += a.Data[rowOffsetA+k] * bT.Data[rowOffsetB+k]
						sum += a.Data[rowOffsetA+k+1] * bT.Data[rowOffsetB+k+1]
						sum += a.Data[rowOffsetA+k+2] * bT.Data[rowOffsetB+k+2]
						sum += a.Data[rowOffsetA+k+3] * bT.Data[rowOffsetB+k+3]
					}
					for ; k < K; k++ {
						sum += a.Data[rowOffsetA+k] * bT.Data[rowOffsetB+k]
					}
					out[i*N+j] = sum
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	res := NewTensor(out, []int{M, N}, a.ReqGrad || b.ReqGrad)
	res.Op = MatMulOp{}
	res.Parents = []*Tensor{a, b}
	return res
}
