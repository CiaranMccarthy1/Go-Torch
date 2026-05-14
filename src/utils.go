package gotorch

import "math"

func Transpose(t *Tensor) *Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	backend := resolveBackend(t)
	out := backend.Transpose(t.ensureStorage(), rows, cols)
	return newTensorFromStorage(out, []int{cols, rows}, t.ReqGrad, backend)
}

func ClipGradNorm(tensors []*Tensor, maxNorm float32) float32 {
	if maxNorm <= 0 {
		panic("maxNorm must be positive")
	}

	var sum float64
	for _, t := range tensors {
		if t == nil || t.gradStorage == nil {
			continue
		}
		gradData := t.gradStorage.Data()
		for _, g := range gradData {
			if !isFinite32(g) {
				panic("non-finite gradient in ClipGradNorm")
			}
			sum += float64(g * g)
		}
	}

	if math.IsNaN(sum) || math.IsInf(sum, 0) {
		panic("non-finite norm in ClipGradNorm")
	}

	norm := float32(math.Sqrt(sum))
	if norm == 0 || norm <= maxNorm {
		return norm
	}
	if !isFinite32(norm) {
		panic("non-finite norm in ClipGradNorm")
	}

	scale := maxNorm / norm
	if !isFinite32(scale) {
		panic("non-finite scale in ClipGradNorm")
	}

	for _, t := range tensors {
		if t == nil || t.gradStorage == nil {
			continue
		}
		gradData := t.gradStorage.Data()
		for i, g := range gradData {
			gradData[i] = g * scale
		}
	}

	return norm
}
