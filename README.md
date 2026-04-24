# Go-Torch — Deep Learning Framework in Go

A zero-dependency reverse-mode autograd engine and tensor library built 
from scratch in Go, with a storage-first backend architecture and 
cache-aware concurrent matrix operations.

## What makes this interesting

Most deep learning frameworks delegate the hard parts to C extensions. 
Go-Torch implements the full autograd graph, gradient accumulation, and 
backend dispatch entirely in pure Go — no cgo, no bindings. The 
storage-first backend interface decouples tensor data from compute, 
making it possible to swap CPU for WebGPU or any future accelerator 
without changing operation signatures. Cache-aware block MatMul with 
row/column partitioned goroutines reduces write contention on the 
backward pass, which is where naive implementations typically collapse 
under load.

## Architecture

`Tensor` wraps a `TensorStorage` struct rather than raw slices, keeping 
data and gradient memory in the same allocation unit. Host access is 
explicit via `Data()` and `Grad()`, and gradient injection from external 
sources uses `SetGrad(...)` — a deliberate boundary that makes the 
forward/backward interface unambiguous.

The backend interface owns all compute: `MatMul`, `Transpose`, `ReLU`, 
`Embed`, in-place accumulation, and the Adam optimiser step all dispatch 
through it. The CPU backend uses block-tiled loops with partitioned 
goroutines to improve L1/L2 cache locality. A WebGPU backend is 
scaffolded behind an availability guard — `NewWebGPUBackend()` returns 
a clear error explaining the Dawn/wgpu-native bridge requirement, so the 
pure-Go build stays clean.

Operations implemented: Matrix Multiplication (forward + backward), 
Transpose, ReLU, Embeddings, Hierarchical Softmax.

## Performance

Benchmarks run on CPU backend (`go test ./src -run ^$ -bench BenchmarkMatMul -benchmem`):

| Operation | Size | ns/op | B/op | allocs/op |
|---|---|---:|---:|---:|
| MatMul Forward | 128×128×128 | 571,642 | 131,876 | 25 |
| MatMul Forward | 256×256×256 | 3,830,848 | 525,060 | 25 |
| MatMul Backward | 128×128×128 | 1,982,238 | 461,009 | 74 |
| MatMul Backward | 256×256×256 | 12,856,377 | 1,842,733 | 74 |

[BENCHMARK NEEDED: end-to-end training comparison against a equivalent 
NumPy MLP on the same task — e.g. XOR or MNIST subset. Measure 
epochs/sec and final loss convergence. This is the number that belongs 
on your CV.]

## Usage

```go
package main

import (
    "fmt"
    gt "github.com/CiaranMccarthy1/go-torch/src"
)

func main() {
    A := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)
    B := gt.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, true)

    C := gt.MatMul(A, B)
    fmt.Println(C.Data()) // forward output

    C.SetGrad([]float32{1, 1, 1, 1})
    C.Backward()
    fmt.Println(A.Grad()) // verified against finite differences
}
```

More complete examples in `examples/`.

## Test Coverage

Gradient correctness is validated against finite differences, not just 
unit assertions:

- `TestMatMulBackwardFiniteDifference` — gradient parity with numerical 
  approximation
- `TestMatMulForwardMatchesNaive` — numerical parity with reference impl
- `TestReLUForwardBackward` — backend-dispatched activation and gradient 
  flow
- `TestEmbeddingForwardBackward` — embedding gather and weight gradient 
  accumulation
- `TestCPUBackendDefault` — backend selection regression

```bash
go test ./...
```

## Installation

```bash
go get github.com/CiaranMccarthy1/go-torch
```

## Stack

Go 1.21+ · Pure standard library · No cgo · No external dependencies
