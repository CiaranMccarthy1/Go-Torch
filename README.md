# go-torch

[![Go](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml/badge.svg)](https://github.com/CiaranMccarthy1/go-torch/actions/workflows/go.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight deep learning framework in Go, supporting autograd and efficient tensor operations.

## Features

- **Autograd**: Automatic differentiation engine.
- **Tensors**: N-dimensional arrays with gradient support.
- **Backend Abstraction**: Storage-first backend interface with CPU backend as the default.
- **Operations**:
	- Matrix Multiplication (cache-aware, concurrent)
  - Transpose
  - Activations (ReLU)
  - Embeddings
  - Hierarchical Softmax
- **Zero Dependencies**: Pure Go standard library.

## Installation

```bash
go get github.com/CiaranMccarthy1/go-torch
```

## Usage

```go
package main

import (
	"fmt"
	gt "github.com/CiaranMccarthy1/go-torch/src"
)

func main() {
	// Create Tensors
	A := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)
	B := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, true)

	// Multiply
	C := gt.MatMul(A, B)
	fmt.Println(C.Data())

	// Backprop
	C.SetGrad([]float32{1, 1, 1, 1})
	C.Backward()

	fmt.Println(A.Grad())
}
```

## Structure

- `tensor.go`: Core data structure.
- `backend.go`: Backend interface, storage abstraction, and CPU backend implementation.
- `webgpu_backend.go`: WebGPU backend scaffold and availability guard.
- `autograd.go`: Backward pass engine.
- `ops_*.go`: Mathematical operations and layers.
- `examples/`: runnable examples.

## Architecture Notes

- `Tensor` stores data and gradients in `TensorStorage` (`storage` and `gradStorage`) instead of raw slices.
- Host access is explicit through `Data()` and `Grad()`, while host-to-device gradient injection uses `SetGrad(...)`.
- Backend interfaces now use `TensorStorage` end to end (`MatMul`, `Transpose`, `ReLU`, `Embed`, in-place accumulation, and Adam step).
- CPU `MatMul` forward and backward use block-based loops and row/column partitioned workers to improve cache locality and reduce write contention.

## WebGPU Note

- `WebGPUBackend` is scaffolded but intentionally unavailable in this pure-Go build.
- `NewWebGPUBackend()` currently returns an error explaining that cgo and a native WebGPU bridge (Dawn or wgpu-native) are required.

## Regression Coverage

- `TestCPUBackendDefault`: verifies CPU remains the default backend.
- `TestMatMulForwardMatchesNaive`: checks numerical parity with a naive implementation.
- `TestMatMulBackwardFiniteDifference`: validates gradients against finite differences.
- `TestReLUForwardBackward`: validates backend-dispatched ReLU forward and gradient flow.
- `TestEmbeddingForwardBackward`: validates embedding gather and weight gradient accumulation.

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
| `BenchmarkMatMulForward_128x128x128-8` | 424145 | 131840 | 25 |
| `BenchmarkMatMulForward_256x256x256-8` | 2682502 | 525059 | 25 |
| `BenchmarkMatMulBackward_128x128x128-8` | 1227620 | 460968 | 74 |
| `BenchmarkMatMulBackward_256x256x256-8` | 8545213 | 1840809 | 74 |

## Contributing

Pull requests are welcome!