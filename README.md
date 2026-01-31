# go-torch

[![Go](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml/badge.svg)](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight deep learning framework in Go, supporting autograd and efficient tensor operations.

## Features

- **Autograd**: Automatic differentiation engine.
- **Tensors**: N-dimensional arrays with gradient support.
- **Operations**:
  - Matrix Multiplication (concurrent)
  - Transpose
  - Activations (ReLU)
  - Embeddings
  - Hierarchical Softmax
- **Zero Dependencies**: Pure Go standard library (uses `unsafe` and `sync/atomic` for performance).

## Installation

```bash
go get github.com/CiaranMccarthy1/go-torch
```

## Usage

```go
package main

import (
	"fmt"
	gt "github.com/CiaranMccarthy1/go-torch"
)

func main() {
	// Create Tensors
	A := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)
	B := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, true)

	// Multiply
	C := gt.MatMul(A, B)

	// Backprop
	C.Grad = []float32{1, 1, 1, 1}
	C.Backward()

	fmt.Println(A.Grad)
}
```

## Structure

- `tensor.go`: Core data structure.
- `autograd.go`: Backward pass engine.
- `ops_*.go`: Mathematical operations and layers.
- `examples/`: runnable examples.

## Contributing

Pull requests are welcome!