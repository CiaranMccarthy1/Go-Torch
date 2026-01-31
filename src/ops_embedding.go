package gotorch

type EmbeddingOp struct{}

func (op EmbeddingOp) Backward(t *Tensor) {
	weights, indices := t.Parents[0], t.Parents[1]
	dim := weights.Shape[1]
	if weights.ReqGrad {
		for i, idxFloat := range indices.Data {
			idx := int(idxFloat)
			for k := 0; k < dim; k++ {
				AtomicAddFloat32(&weights.Grad[idx*dim+k], t.Grad[i*dim+k])
			}
		}
	}
}

func Embed(weights, indices *Tensor) *Tensor {
	batch, dim := indices.Shape[0], weights.Shape[1]
	out := make([]float32, batch*dim)
	for i, idxFloat := range indices.Data {
		idx := int(idxFloat)
		copy(out[i*dim:(i+1)*dim], weights.Data[idx*dim:(idx+1)*dim])
	}
	res := NewTensor(out, []int{batch, dim}, weights.ReqGrad)
	res.Op = EmbeddingOp{}
	res.Parents = []*Tensor{weights, indices}
	return res
}
