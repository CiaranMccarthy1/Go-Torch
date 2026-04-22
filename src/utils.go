package gotorch

func Transpose(t *Tensor) *Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	backend := resolveBackend(t)
	out := backend.Transpose(t.ensureStorage(), rows, cols)
	return newTensorFromStorage(out, []int{cols, rows}, t.ReqGrad, backend)
}
