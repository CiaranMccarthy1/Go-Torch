package main

import (
	"fmt"

	gt "github.com/CiaranMccarthy1/go-torch/src"
)

func main() {
	fmt.Println("Running go-torch simple example...")

	// 1. Define Tensors
	// A: 2x3 matrix
	dataA := []float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	}
	A := gt.NewTensor(dataA, []int{2, 3}, true)

	// B: 3x2 matrix
	dataB := []float32{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
	}
	B := gt.NewTensor(dataB, []int{3, 2}, true)

	// 2. Perform Operation (MatMul)
	// C = A * B (should be 2x2)
	C := gt.MatMul(A, B)

	fmt.Printf("Result Shape: %v\n", C.Shape)
	fmt.Printf("Result Data: %v\n", C.Data())

	// 3. Backward Pass
	// Let's suppose we want to maximize the sum of C elements (gradient = 1 everywhere)
	// Or we can just call Backward() which defaults to gradient 1.0 for scalar,
	// but here C is 2x2. C.Backward() handles it if we treat it as sum?
	// The implementation of Backward says: if no gradient is set, it seeds with [1.0].
	// This only works if C is scalar or if we manually set Grad.
	// Since C is 2x2, let's create a scalar loss = Sum(C) explicitly by some mechanism
	// or just manually set gradient to ones and backprop.

	// Manually set gradient of C to all ones (dL/dC_ij = 1)
	C.SetGrad([]float32{1.0, 1.0, 1.0, 1.0})

	fmt.Println("Backpropagating...")
	C.Backward()

	fmt.Printf("Gradient w.r.t A:\n%v\n", A.Grad())
	fmt.Printf("Gradient w.r.t B:\n%v\n", B.Grad())
}
