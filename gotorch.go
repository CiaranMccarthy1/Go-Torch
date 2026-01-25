package gotorch

import (
	"fmt"
	"math"
	"sync"
)

// ============================================================================
// 1. TENSOR SYSTEM
// ============================================================================

type Tensor struct {
	Data    []float64
	Shape   []int
	Strides []int
	Grad    []float64
	Parents []*Tensor
	Op      Operation
	ReqGrad bool
}

type Operation interface {
	Backward(t *Tensor)
}

// NewTensor creates a new tensor.
// data: Flat array of values. If nil, allocates zeroed memory.
// shape: Dimensions
// reqGrad: Set true if this is a learnable weight (like W or B).
func NewTensor(data []float64, shape []int, reqGrad bool) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	if data == nil {
		data = make([]float64, size)
	} else if len(data) != size {
		panic(fmt.Sprintf("NewTensor: Data size %d does not match shape %v", len(data), shape))
	}

	t := &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: calculateStrides(shape),
		ReqGrad: reqGrad,
	}

	if reqGrad {
		t.Grad = make([]float64, size)
	}
	return t
}

func calculateStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

// ZeroGrad clears gradients. Must be called before every training step.
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	}
}

// String provides a pretty-print format for debugging.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(Shape: %v, ReqGrad: %v, Data: %.4f...)", t.Shape, t.ReqGrad, t.Data[:min(len(t.Data), 5)])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// 2. AUTOGRAD
// ============================================================================

// Backward computes the gradients of all ancestors in the graph.
func (t *Tensor) Backward() {
	if t.Grad == nil {
		t.Grad = []float64{1.0} // Implicit 1.0 gradient for scalar loss
	}

	// Topological Sort to ensure correct dependency order
	order := []*Tensor{}
	visited := make(map[*Tensor]bool)
	var build func(*Tensor)
	build = func(v *Tensor) {
		if !visited[v] {
			visited[v] = true
			for _, p := range v.Parents {
				build(p)
			}
			order = append(order, v)
		}
	}
	build(t)

	// Reverse Pass
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.Op != nil {
			node.Op.Backward(node)
		}
	}
}

// ============================================================================
// 3. LAYERS & OPERATIONS
// ============================================================================

// MatMulOp - Matrix Multiplication
type MatMulOp struct{}

func (op MatMulOp) Backward(t *Tensor) {
	a, b := t.Parents[0], t.Parents[1]
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]

	// Gradients: dA = dOut * B^T, dB = A^T * dOut
	if a.ReqGrad {
		for i := 0; i < M; i++ {
			for k := 0; k < K; k++ {
				sum := 0.0
				for j := 0; j < N; j++ {
					sum += t.Grad[i*N+j] * b.Data[k*N+j]
				}
				a.Grad[i*K+k] += sum
			}
		}
	}
	if b.ReqGrad {
		for k := 0; k < K; k++ {
			for j := 0; j < N; j++ {
				sum := 0.0
				for i := 0; i < M; i++ {
					sum += a.Data[i*K+k] * t.Grad[i*N+j]
				}
				b.Grad[k*N+j] += sum
			}
		}
	}
}

func MatMul(a, b *Tensor) *Tensor {
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("MatMul shape mismatch: %v vs %v", a.Shape, b.Shape))
	}
	M, K, N := a.Shape[0], a.Shape[1], b.Shape[1]
	out := make([]float64, M*N)

	// CPU Parallelism (Goroutines)
	var wg sync.WaitGroup
	workers := 4 // optimize based on CPU cores
	rowsPerWorker := (M + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if start >= M {
			break
		}
		if end > M {
			end = M
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < K; k++ {
						sum += a.Data[i*K+k] * b.Data[k*N+j]
					}
					out[i*N+j] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()

	res := NewTensor(out, []int{M, N}, a.ReqGrad || b.ReqGrad)
	res.Op = MatMulOp{}
	res.Parents = []*Tensor{a, b}
	return res
}

// EmbeddingOp (Lookup Table)
type EmbeddingOp struct{}

func (op EmbeddingOp) Backward(t *Tensor) {
	weights, indices := t.Parents[0], t.Parents[1]
	embedDim := weights.Shape[1]

	if weights.ReqGrad {
		for i, idxFloat := range indices.Data {
			idx := int(idxFloat)
			// Accumulate gradients for the specific word vector
			for k := 0; k < embedDim; k++ {
				weights.Grad[idx*embedDim+k] += t.Grad[i*embedDim+k]
			}
		}
	}
}

func Embed(weights, indices *Tensor) *Tensor {
	batch := indices.Shape[0]
	dim := weights.Shape[1]
	out := make([]float64, batch*dim)

	for i, idxFloat := range indices.Data {
		idx := int(idxFloat)
		for k := 0; k < dim; k++ {
			out[i*dim+k] = weights.Data[idx*dim+k]
		}
	}
	res := NewTensor(out, []int{batch, dim}, weights.ReqGrad)
	res.Op = EmbeddingOp{}
	res.Parents = []*Tensor{weights, indices}
	return res
}

// ReLUOp (Activation) ---
type ReLUOp struct{}

func (op ReLUOp) Backward(t *Tensor) {
	input := t.Parents[0]
	if input.ReqGrad {
		for i, val := range input.Data {
			if val > 0 {
				input.Grad[i] += t.Grad[i]
			}
		}
	}
}

func ReLU(t *Tensor) *Tensor {
	out := make([]float64, len(t.Data))
	for i, v := range t.Data {
		if v > 0 {
			out[i] = v
		}
	}
	res := NewTensor(out, t.Shape, t.ReqGrad)
	res.Op = ReLUOp{}
	res.Parents = []*Tensor{t}
	return res
}

// CrossEntropyOp (Loss Function) ---
type CrossEntropyOp struct{}

func (op CrossEntropyOp) Backward(t *Tensor) {
	logits, target := t.Parents[0], t.Parents[1]
	batch := logits.Shape[0]
	vocab := logits.Shape[1]

	if logits.ReqGrad {
		for i := 0; i < batch; i++ {
			offset := i * vocab
			// 1. Re-calculate Softmax
			maxVal := -1e9
			for j := 0; j < vocab; j++ {
				if logits.Data[offset+j] > maxVal {
					maxVal = logits.Data[offset+j]
				}
			}
			sumExp := 0.0
			exps := make([]float64, vocab)
			for j := 0; j < vocab; j++ {
				exps[j] = math.Exp(logits.Data[offset+j] - maxVal)
				sumExp += exps[j]
			}

			// 2. Gradient = (Prob - 1) for true class, else Prob
			trueClass := int(target.Data[i])
			for j := 0; j < vocab; j++ {
				prob := exps[j] / sumExp
				grad := prob
				if j == trueClass {
					grad -= 1.0
				}
				logits.Grad[offset+j] += grad / float64(batch)
			}
		}
	}
}

func CrossEntropy(logits, target *Tensor) *Tensor {
	batch := logits.Shape[0]
	vocab := logits.Shape[1]
	totalLoss := 0.0

	for i := 0; i < batch; i++ {
		offset := i * vocab
		// Softmax Stability Trick
		maxVal := -1e9
		for j := 0; j < vocab; j++ {
			if logits.Data[offset+j] > maxVal {
				maxVal = logits.Data[offset+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < vocab; j++ {
			sumExp += math.Exp(logits.Data[offset+j] - maxVal)
		}

		trueClass := int(target.Data[i])
		logProb := (logits.Data[offset+trueClass] - maxVal) - math.Log(sumExp)
		totalLoss -= logProb
	}

	res := NewTensor([]float64{totalLoss / float64(batch)}, []int{1}, logits.ReqGrad)
	res.Op = CrossEntropyOp{}
	res.Parents = []*Tensor{logits, target}
	return res
}

// ============================================================================
// HIERARCHICAL SOFTMAX COMPONENTS
// ============================================================================

// HSNode represents an internal node in the Hierarchical Softmax binary tree.
type HSNode struct {
	ID     int
	Weight *Tensor // The learned vector for this specific binary decision
}

// HSPath defines the route from root to a specific word (leaf).
type HSPath struct {
	Nodes      []*HSNode // Sequence of nodes to visit
	Directions []float64 // +1 for Left, -1 for Right (or 1 and 0 for sigmoid targets)
}

// HierarchicalSoftmax manages the tree structure for efficient large-vocab training.
type HierarchicalSoftmax struct {
	Nodes []*HSNode
	Paths map[int]HSPath
}

// NewHierarchicalSoftmax builds a balanced binary tree for the given vocabulary.
func NewHierarchicalSoftmax(vocabSize int, hiddenDim int) *HierarchicalSoftmax {
	numInternalNodes := vocabSize - 1
	hs := &HierarchicalSoftmax{
		Nodes: make([]*HSNode, numInternalNodes),
		Paths: make(map[int]HSPath),
	}

	// Initialize all internal decision nodes with weights
	for i := 0; i < numInternalNodes; i++ {
		hs.Nodes[i] = &HSNode{
			ID:     i,
			Weight: NewTensor(nil, []int{1, hiddenDim}, true),
		}
	}

	// Build a balanced binary tree mapping: WordID -> Path
	// This uses a standard binary heap-style indexing for simplicity
	for wordID := 0; wordID < vocabSize; wordID++ {
		path := HSPath{}
		// Current "leaf" position in a complete binary tree
		curr := wordID + vocabSize - 1

		for curr > 0 {
			parent := (curr - 1) / 2
			path.Nodes = append([]*HSNode{hs.Nodes[parent]}, path.Nodes...)

			// If curr is left child (odd), target is 1. If right (even), target is 0.
			if curr%2 == 1 {
				path.Directions = append([]float64{1.0}, path.Directions...)
			} else {
				path.Directions = append([]float64{0.0}, path.Directions...)
			}
			curr = parent
		}
		hs.Paths[wordID] = path
	}
	return hs
}

// --- HS Loss Operation ---

type HSLossOp struct {
	HS     *HierarchicalSoftmax
	Target int
}

func (op HSLossOp) Backward(t *Tensor) {
	hidden := t.Parents[0]
	path := op.HS.Paths[op.Target]

	if hidden.ReqGrad {
		for i, node := range path.Nodes {
			// Re-calculate sigmoid during backward
			// P(left) = sigmoid(node.Weight @ hidden)
			score := 0.0
			for k := range hidden.Data {
				score += node.Weight.Data[k] * hidden.Data[k]
			}
			prob := 1.0 / (1.0 + math.Exp(-score))

			// Gradient of Log-Sigmoid is (Target - Prob)
			grad := prob - path.Directions[i]

			// Update Node Weights
			for k := range node.Weight.Data {
				node.Weight.Grad[k] += grad * hidden.Data[k]
				// Pass gradient back to hidden state
				hidden.Grad[k] += grad * node.Weight.Data[k]
			}
		}
	}
}

// HSLoss calculates the Negative Log Likelihood using the Hierarchical tree.
// Complexity: O(log VocabSize) instead of O(VocabSize).
func HSLoss(hs *HierarchicalSoftmax, hidden *Tensor, targetID int) *Tensor {
	path := hs.Paths[targetID]
	totalLoss := 0.0

	for i, node := range path.Nodes {
		// dot product: node.Weight @ hidden
		score := 0.0
		for k := range hidden.Data {
			score += node.Weight.Data[k] * hidden.Data[k]
		}

		// Sigmoid: 1 / (1 + exp(-score))
		prob := 1.0 / (1.0 + math.Exp(-score))

		// Cross Entropy for this binary decision
		target := path.Directions[i]
		if target == 1.0 {
			totalLoss -= math.Log(prob + 1e-9)
		} else {
			totalLoss -= math.Log(1.0 - prob + 1e-9)
		}
	}

	res := NewTensor([]float64{totalLoss}, []int{1}, hidden.ReqGrad)
	res.Op = HSLossOp{HS: hs, Target: targetID}
	res.Parents = []*Tensor{hidden}
	return res
}
