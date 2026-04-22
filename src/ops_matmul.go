package gotorch

import (
	"fmt"
)

type MatMulOp struct{}

func (op MatMulOp) Backward(t *Tensor) {
	a, b := t.Parents[0], t.Parents[1]
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	backend := resolveBackend(t, a, b)

	gradA, gradB := backend.MatMulBackward(a.values(), b.values(), t.Grad, M, K, N, a.ReqGrad, b.ReqGrad)

	if a.ReqGrad {
		if a.Grad == nil {
			a.Grad = make([]float32, M*K)
		}
		for i, v := range gradA {
			a.Grad[i] += v
		}
	}

	if b.ReqGrad {
		if b.Grad == nil {
			b.Grad = make([]float32, K*N)
		}
		for i, v := range gradB {
			b.Grad[i] += v
		}
	}
}

func MatMul(a, b *Tensor) *Tensor {
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("Shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	backend := resolveBackend(a, b)
	out := backend.MatMulForward(a.values(), b.values(), M, K, N)

	res := NewTensorWithBackend(out, []int{M, N}, a.ReqGrad || b.ReqGrad, backend)
	res.Op = MatMulOp{}
	res.Parents = []*Tensor{a, b}
	return res
}
