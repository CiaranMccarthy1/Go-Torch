package gotorch

type EmbeddingOp struct{}

func (op EmbeddingOp) Backward(t *Tensor) {
	weights, indices := t.Parents[0], t.Parents[1]
	dim := weights.Shape[1]
	if !weights.ReqGrad || t.gradStorage == nil {
		return
	}

	backend := resolveBackend(weights, indices, t)
	if weights.gradStorage == nil {
		weights.gradStorage = backend.ZeroStorage(shapeSize(weights.Shape))
	}

	batch := indices.Shape[0]
	gradWeights := backend.EmbedBackward(weights.ensureStorage(), indices.ensureStorage(), t.gradStorage, batch, dim)
	backend.AddInPlace(weights.gradStorage, gradWeights)
}

func Embed(weights, indices *Tensor) *Tensor {
	batch, dim := indices.Shape[0], weights.Shape[1]
	backend := resolveBackend(weights, indices)
	out := backend.EmbedForward(weights.ensureStorage(), indices.ensureStorage(), batch, dim)
	res := newTensorFromStorage(out, []int{batch, dim}, weights.ReqGrad, backend)
	res.Op = EmbeddingOp{}
	res.Parents = []*Tensor{weights, indices}
	return res
}
