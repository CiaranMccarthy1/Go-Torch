package gotorch_test

import (
	"testing"

	gotorch "github.com/CiaranMccarthy1/go-torch/src"
)

func benchmarkMatMulForward(b *testing.B, m, k, n int) {
	a := gotorch.NewTensor(randomData(m*k, 7), []int{m, k}, false)
	weights := gotorch.NewTensor(randomData(k*n, 13), []int{k, n}, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := gotorch.MatMul(a, weights)
		if len(out.Data()) != m*n {
			b.Fatalf("unexpected output size: got=%d want=%d", len(out.Data()), m*n)
		}
	}
}

func benchmarkMatMulBackward(b *testing.B, m, k, n int) {
	a := gotorch.NewTensor(randomData(m*k, 17), []int{m, k}, true)
	weights := gotorch.NewTensor(randomData(k*n, 19), []int{k, n}, true)
	upstream := randomData(m*n, 23)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.ZeroGrad()
		weights.ZeroGrad()

		out := gotorch.MatMul(a, weights)
		out.SetGrad(append([]float32(nil), upstream...))
		out.Backward()
	}
}

func BenchmarkMatMulForward_128x128x128(b *testing.B) {
	benchmarkMatMulForward(b, 128, 128, 128)
}

func BenchmarkMatMulForward_256x256x256(b *testing.B) {
	benchmarkMatMulForward(b, 256, 256, 256)
}

func BenchmarkMatMulBackward_128x128x128(b *testing.B) {
	benchmarkMatMulBackward(b, 128, 128, 128)
}

func BenchmarkMatMulBackward_256x256x256(b *testing.B) {
	benchmarkMatMulBackward(b, 256, 256, 256)
}
