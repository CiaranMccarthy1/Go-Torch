package gotorch

import "testing"

func TestReLUForwardBackward(t *testing.T) {
	input := NewTensor([]float32{-2, -0.5, 0, 1, 3}, []int{5}, true)
	out := ReLU(input)

	expectedForward := []float32{0, 0, 0, 1, 3}
	requireCloseSlice(t, "relu forward", out.Data(), expectedForward, 1e-6)

	out.SetGrad([]float32{1, 2, 3, 4, 5})
	out.Backward()

	expectedGrad := []float32{0, 0, 0, 4, 5}
	requireCloseSlice(t, "relu backward", input.Grad(), expectedGrad, 1e-6)
}

func TestEmbeddingForwardBackward(t *testing.T) {
	weights := NewTensor([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, []int{3, 2}, true)
	indices := NewTensor([]float32{2, 0, 2}, []int{3}, false)

	out := Embed(weights, indices)
	expectedForward := []float32{
		5, 6,
		1, 2,
		5, 6,
	}
	requireCloseSlice(t, "embed forward", out.Data(), expectedForward, 1e-6)

	out.SetGrad([]float32{
		1, 1,
		2, 2,
		3, 3,
	})
	out.Backward()

	expectedGrad := []float32{
		2, 2,
		0, 0,
		4, 4,
	}
	requireCloseSlice(t, "embed backward", weights.Grad(), expectedGrad, 1e-6)
}
