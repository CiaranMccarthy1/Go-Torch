package gotorch

type ReLUOp struct{}

func (op ReLUOp) Backward(t *Tensor) {
	input := t.Parents[0]
	if input.ReqGrad {
		for i, val := range input.Data {
			if val > 0 {
				AtomicAddFloat32(&input.Grad[i], t.Grad[i])
			}
		}
	}
}

func ReLU(t *Tensor) *Tensor {
	out := make([]float32, len(t.Data))
	for i, v := range t.Data {
		if v > 0 {
			out[i] = v
		}
	}
	res := NewTensor(out, t.Shape, t.ReqGrad)
	res.Op = ReLUOp{}
	res.Parents = []*Tensor{t}
	return res
}
