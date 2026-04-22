package gotorch

import (
	"fmt"
)

type MatMulOp struct{}

func (op MatMulOp) Backward(t *Tensor) {
	a, b := t.Parents[0], t.Parents[1]
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	backend := resolveBackend(t, a, b)
	if t.gradStorage == nil {
		return
	}

	gradA, gradB := backend.MatMulBackward(a.ensureStorage(), b.ensureStorage(), t.gradStorage, M, K, N, a.ReqGrad, b.ReqGrad)

	if a.ReqGrad {
		if a.gradStorage == nil {
			a.gradStorage = backend.ZeroStorage(M * K)
		}
		backend.AddInPlace(a.gradStorage, gradA)
	}

	if b.ReqGrad {
		if b.gradStorage == nil {
			b.gradStorage = backend.ZeroStorage(K * N)
		}
		backend.AddInPlace(b.gradStorage, gradB)
	}
}

func MatMul(a, b *Tensor) *Tensor {
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("Shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	backend := resolveBackend(a, b)
	out := backend.MatMulForward(a.ensureStorage(), b.ensureStorage(), M, K, N)

	res := newTensorFromStorage(out, []int{M, N}, a.ReqGrad || b.ReqGrad, backend)
	res.Op = MatMulOp{}
	res.Parents = []*Tensor{a, b}
	return res
}
