package gotorch

import (
	"math"
	"testing"
)

func TestClipGradNormScales(t *testing.T) {
	param := NewTensor(nil, []int{2, 2}, true)
	param.SetGrad([]float32{3, 4, 0, 0})

	norm := ClipGradNorm([]*Tensor{param}, 2)
	if math.Abs(float64(norm-5)) > 1e-6 {
		t.Fatalf("unexpected norm: got=%f want=5", norm)
	}

	requireCloseSlice(t, "clip grad", param.Grad(), []float32{1.2, 1.6, 0, 0}, 1e-6)
}
