package gotorch

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

type Tensor struct {
	Data    []float64
	Shape   []int
	Grad    []float64
	Parents []*Tensor
	Op      Operation
	ReqGrad bool
}

type Operation interface {
	Backward(t *Tensor)
}

func NewTensor(data []float64, shape []int, reqGrad bool) *Tensor {
	size := 1
	for _, dim := range shape { size *= dim }
	if data == nil { data = make([]float64, size) }
	t := &Tensor{Data: data, Shape: shape, ReqGrad: reqGrad}
	if reqGrad { t.Grad = make([]float64, size) }
	return t
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad { t.Grad[i] = 0 }
	}
}

func (t *Tensor) Backward() {
	if t.Grad == nil { t.Grad = []float64{1.0} }
	order := []*Tensor{}
	visited := make(map[*Tensor]bool)
	var build func(*Tensor)
	build = func(v *Tensor) {
		if !visited[v] {
			visited[v] = true
			for _, p := range v.Parents { build(p) }
			order = append(order, v)
		}
	}
	build(t)
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.Op != nil { node.Op.Backward(node) }
	}
}

// --- Thread-Safe Operations ---

type MatMulOp struct{}

func (op MatMulOp) Backward(t *Tensor) {
	a, b := t.Parents[0], t.Parents[1]
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]

	if a.ReqGrad {
		for i := 0; i < M; i++ {
			for k := 0; k < K; k++ {
				var sum float64
				for j := 0; j < N; j++ {
					sum += t.Grad[i*N+j] * b.Data[k*N+j]
				}
				atomic.AddFloat64(&a.Grad[i*K+k], sum) // Thread-safe
			}
		}
	}
	if b.ReqGrad {
		for k := 0; k < K; k++ {
			for j := 0; j < N; j++ {
				var sum float64
				for i := 0; i < M; i++ {
					sum += a.Data[i*K+k] * t.Grad[i*N+j]
				}
				atomic.AddFloat64(&b.Grad[k*N+j], sum) // Thread-safe
			}
		}
	}
}

func MatMul(a, b *Tensor) *Tensor {
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	out := make([]float64, M*N)
	var wg sync.WaitGroup
	// Parallelize rows
	for i := 0; i < M; i++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			for j := 0; j < N; j++ {
				var sum float64
				for k := 0; k < K; k++ {
					sum += a.Data[r*K+k] * b.Data[k*N+j]
				}
				out[r*N+j] = sum
			}
		}(i)
	}
	wg.Wait()
	res := NewTensor(out, []int{M, N}, a.ReqGrad || b.ReqGrad)
	res.Op = MatMulOp{}; res.Parents = []*Tensor{a, b}
	return res
}

type EmbeddingOp struct{}

func (op EmbeddingOp) Backward(t *Tensor) {
	weights, indices := t.Parents[0], t.Parents[1]
	dim := weights.Shape[1]
	if weights.ReqGrad {
		for i, idxFloat := range indices.Data {
			idx := int(idxFloat)
			for k := 0; k < dim; k++ {
				atomic.AddFloat64(&weights.Grad[idx*dim+k], t.Grad[i*dim+k])
			}
		}
	}
}

func Embed(weights, indices *Tensor) *Tensor {
	batch, dim := indices.Shape[0], weights.Shape[1]
	out := make([]float64, batch*dim)
	for i, idxFloat := range indices.Data {
		idx := int(idxFloat)
		copy(out[i*dim:(i+1)*dim], weights.Data[idx*dim:(idx+1)*dim])
	}
	res := NewTensor(out, []int{batch, dim}, weights.ReqGrad)
	res.Op = EmbeddingOp{}; res.Parents = []*Tensor{weights, indices}
	return res
}

type ReLUOp struct{}

func (op ReLUOp) Backward(t *Tensor) {
	input := t.Parents[0]
	if input.ReqGrad {
		for i, val := range input.Data {
			if val > 0 { atomic.AddFloat64(&input.Grad[i], t.Grad[i]) }
		}
	}
}

func ReLU(t *Tensor) *Tensor {
	out := make([]float64, len(t.Data))
	for i, v := range t.Data { if v > 0 { out[i] = v } }
	res := NewTensor(out, t.Shape, t.ReqGrad)
	res.Op = ReLUOp{}; res.Parents = []*Tensor{t}
	return res
}

// --- Hierarchical Softmax ---

type HSNode struct { ID int; Weight *Tensor }
type HSPath struct { Nodes []*HSNode; Directions []float64 }
type HierarchicalSoftmax struct { Nodes []*HSNode; Paths map[int]HSPath }

func NewHierarchicalSoftmax(vocabSize, hiddenDim int) *HierarchicalSoftmax {
	hs := &HierarchicalSoftmax{Nodes: make([]*HSNode, vocabSize-1), Paths: make(map[int]HSPath)}
	for i := 0; i < vocabSize-1; i++ {
		hs.Nodes[i] = &HSNode{ID: i, Weight: NewTensor(nil, []int{1, hiddenDim}, true)}
	}
	for wordID := 0; wordID < vocabSize; wordID++ {
		path := HSPath{}; curr := wordID + vocabSize - 1
		for curr > 0 {
			parent := (curr - 1) / 2
			path.Nodes = append([]*HSNode{hs.Nodes[parent]}, path.Nodes...)
			dir := 0.0; if curr%2 == 1 { dir = 1.0 }
			path.Directions = append([]float64{dir}, path.Directions...)
			curr = parent
		}
		hs.Paths[wordID] = path
	}
	return hs
}

type HSLossOp struct { HS *HierarchicalSoftmax; Target int }

func (op HSLossOp) Backward(t *Tensor) {
	hidden := t.Parents[0]
	path := op.HS.Paths[op.Target]
	for i, node := range path.Nodes {
		var score float64
		for k := range hidden.Data { score += node.Weight.Data[k] * hidden.Data[k] }
		prob := 1.0 / (1.0 + math.Exp(-score))
		grad := prob - path.Directions[i]
		for k := range node.Weight.Data {
			atomic.AddFloat64(&node.Weight.Grad[k], grad * hidden.Data[k])
			atomic.AddFloat64(&hidden.Grad[k], grad * node.Weight.Data[k])
		}
	}
}

func HSLoss(hs *HierarchicalSoftmax, hidden *Tensor, targetID int) *Tensor {
	path := hs.Paths[targetID]; loss := 0.0
	for i, node := range path.Nodes {
		var score float64
		for k := range hidden.Data { score += node.Weight.Data[k] * hidden.Data[k] }
		prob := 1.0 / (1.0 + math.Exp(-score))
		if path.Directions[i] == 1.0 { loss -= math.Log(prob + 1e-9) } else { loss -= math.Log(1.0 - prob + 1e-9) }
	}
	res := NewTensor([]float64{loss}, []int{1}, true)
	res.Op = HSLossOp{HS: hs, Target: targetID}; res.Parents = []*Tensor{hidden}
	return res
}
