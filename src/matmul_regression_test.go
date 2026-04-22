package gotorch

import (
	"math"
	"math/rand"
	"testing"
)

func randomData(size int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, size)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

func naiveMatMul(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for x := 0; x < k; x++ {
				sum += a[i*k+x] * b[x*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return out
}

func weightedSum(x, w []float32) float32 {
	var out float32
	for i := range x {
		out += x[i] * w[i]
	}
	return out
}

func requireCloseSlice(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > tol {
			t.Fatalf("%s mismatch at idx=%d: got=%f want=%f diff=%f tol=%f", name, i, got[i], want[i], diff, tol)
		}
	}
}

func TestCPUBackendDefault(t *testing.T) {
	tensor := NewTensor([]float32{1}, []int{1}, false)
	if tensor.Backend == nil {
		t.Fatal("expected default backend to be set")
	}
	if tensor.Backend.Name() != "cpu" {
		t.Fatalf("expected default backend cpu, got=%s", tensor.Backend.Name())
	}
}

func TestMatMulForwardMatchesNaive(t *testing.T) {
	m, k, n := 32, 48, 24
	aData := randomData(m*k, 11)
	bData := randomData(k*n, 29)

	a := NewTensor(append([]float32(nil), aData...), []int{m, k}, false)
	b := NewTensor(append([]float32(nil), bData...), []int{k, n}, false)
	out := MatMul(a, b)

	if out.Shape[0] != m || out.Shape[1] != n {
		t.Fatalf("unexpected output shape: got=%v want=[%d %d]", out.Shape, m, n)
	}

	expected := naiveMatMul(aData, bData, m, k, n)
	requireCloseSlice(t, "matmul forward", out.Data, expected, 1e-4)
}

func TestMatMulBackwardFiniteDifference(t *testing.T) {
	const eps float32 = 1e-3
	m, k, n := 3, 4, 2
	aData := randomData(m*k, 101)
	bData := randomData(k*n, 202)
	upstream := randomData(m*n, 303)

	a := NewTensor(append([]float32(nil), aData...), []int{m, k}, true)
	b := NewTensor(append([]float32(nil), bData...), []int{k, n}, true)
	out := MatMul(a, b)
	out.Grad = append([]float32(nil), upstream...)
	out.Backward()

	numGradA := make([]float32, len(aData))
	numGradB := make([]float32, len(bData))

	aProbe := append([]float32(nil), aData...)
	for idx := range aProbe {
		orig := aProbe[idx]
		aProbe[idx] = orig + eps
		plus := weightedSum(naiveMatMul(aProbe, bData, m, k, n), upstream)
		aProbe[idx] = orig - eps
		minus := weightedSum(naiveMatMul(aProbe, bData, m, k, n), upstream)
		numGradA[idx] = (plus - minus) / (2 * eps)
		aProbe[idx] = orig
	}

	bProbe := append([]float32(nil), bData...)
	for idx := range bProbe {
		orig := bProbe[idx]
		bProbe[idx] = orig + eps
		plus := weightedSum(naiveMatMul(aData, bProbe, m, k, n), upstream)
		bProbe[idx] = orig - eps
		minus := weightedSum(naiveMatMul(aData, bProbe, m, k, n), upstream)
		numGradB[idx] = (plus - minus) / (2 * eps)
		bProbe[idx] = orig
	}

	requireCloseSlice(t, "matmul grad A", a.Grad, numGradA, 2e-2)
	requireCloseSlice(t, "matmul grad B", b.Grad, numGradB, 2e-2)
}
