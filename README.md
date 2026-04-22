# go-torch

[![Go](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml/badge.svg)](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight deep learning framework in Go, supporting autograd and efficient tensor operations.

## Features

- **Autograd**: Automatic differentiation engine.
- **Tensors**: N-dimensional arrays with gradient support.
- **Backend Abstraction**: Pluggable backend interface with CPU backend as the default.
- **Operations**:
	- Matrix Multiplication (cache-aware, concurrent)
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
- `backend.go`: Backend interface, storage abstraction, and CPU backend implementation.
- `autograd.go`: Backward pass engine.
- `ops_*.go`: Mathematical operations and layers.
- `examples/`: runnable examples.

## Architecture Notes

- `Tensor` keeps the existing public fields (`Data`, `Shape`, `Grad`, etc.) for backward compatibility.
- Tensor allocation now routes through a `Backend` interface, with `CPUBackend` configured as default.
- Operation dispatch for `MatMul` and `Transpose` is now backend-driven, so alternate backends can be added without changing public APIs.
- CPU `MatMul` forward and backward use block-based loops and row/column partitioned workers to improve cache locality and reduce write contention.

## Regression Coverage

- `TestCPUBackendDefault`: verifies CPU remains the default backend.
- `TestMatMulForwardMatchesNaive`: checks numerical parity with a naive implementation.
- `TestMatMulBackwardFiniteDifference`: validates gradients against finite differences.

Run with:

```bash
go test ./...
```

## Benchmarks

Command used:

```bash
go test ./src -run ^$ -bench BenchmarkMatMul -benchmem
```

Latest local results:

| Benchmark | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| `BenchmarkMatMulForward_128x128x128-8` | 415092 | 131849 | 24 |
| `BenchmarkMatMulForward_256x256x256-8` | 3491172 | 525067 | 24 |
| `BenchmarkMatMulBackward_128x128x128-8` | 1849093 | 460709 | 68 |
| `BenchmarkMatMulBackward_256x256x256-8` | 9152327 | 1836979 | 68 |

## Contributing

Pull requests are welcome!