package gotorch

type ReLUOp struct{}

func (op ReLUOp) Backward(t *Tensor) {
	input := t.Parents[0]
	if !input.ReqGrad || t.gradStorage == nil {
		return
	}

	backend := resolveBackend(input, t)
	if input.gradStorage == nil {
		input.gradStorage = backend.ZeroStorage(shapeSize(input.Shape))
	}

	grad := backend.ReLUBackward(input.ensureStorage(), t.gradStorage)
	backend.AddInPlace(input.gradStorage, grad)
}

func ReLU(t *Tensor) *Tensor {
	backend := resolveBackend(t)
	out := backend.ReLUForward(t.ensureStorage())
	res := newTensorFromStorage(out, t.Shape, t.ReqGrad, backend)
	res.Op = ReLUOp{}
	res.Parents = []*Tensor{t}
	return res
}
