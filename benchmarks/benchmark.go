package main

import (
	"fmt"
	"math/rand"
	"testing"

	gotorch "github.com/CiaranMccarthy1/go-torch/src"
)

func randomData(size int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, size)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

func benchmarkMatMulSize(b *testing.B, size int) {
	m, k, n := size, size, size
	a := gotorch.NewTensor(randomData(m*k, int64(11+size)), []int{m, k}, false)
	weights := gotorch.NewTensor(randomData(k*n, int64(13+size)), []int{k, n}, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := gotorch.MatMul(a, weights)
		if len(out.Data()) != m*n {
			b.Fatalf("unexpected output size: got=%d want=%d", len(out.Data()), m*n)
		}
	}
}

func main() {
	sizes := []int{32, 64, 128, 256, 512}
	for _, size := range sizes {
		result := testing.Benchmark(func(b *testing.B) {
			benchmarkMatMulSize(b, size)
		})
		fmt.Printf("BenchmarkMatMul %dx%dx%d: %s\n", size, size, size, result.String())
		fmt.Printf("allocs/op: %d\n", result.AllocsPerOp())
		fmt.Printf("ns/op: %d\n\n", result.NsPerOp())
	}
}
