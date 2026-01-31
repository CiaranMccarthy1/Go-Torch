package gotorch

import "math"

type HSNode struct {
	ID     int
	Weight *Tensor
}
type HSPath struct {
	Nodes      []*HSNode
	Directions []float32
}
type HierarchicalSoftmax struct {
	Nodes []*HSNode
	Paths map[int]HSPath
}

func NewHierarchicalSoftmax(vocabSize, hiddenDim int) *HierarchicalSoftmax {
	hs := &HierarchicalSoftmax{Nodes: make([]*HSNode, vocabSize-1), Paths: make(map[int]HSPath)}
	for i := 0; i < vocabSize-1; i++ {
		hs.Nodes[i] = &HSNode{ID: i, Weight: NewTensor(nil, []int{1, hiddenDim}, true)}
	}
	for wordID := 0; wordID < vocabSize; wordID++ {
		path := HSPath{}
		curr := wordID + vocabSize - 1
		for curr > 0 {
			parent := (curr - 1) / 2
			path.Nodes = append([]*HSNode{hs.Nodes[parent]}, path.Nodes...)
			dir := float32(0.0)
			if curr%2 == 1 {
				dir = 1.0
			}
			path.Directions = append([]float32{dir}, path.Directions...)
			curr = parent
		}
		hs.Paths[wordID] = path
	}
	return hs
}

type HSLossOp struct {
	HS     *HierarchicalSoftmax
	Target int
}

func (op HSLossOp) Backward(t *Tensor) {
	hidden := t.Parents[0]
	path := op.HS.Paths[op.Target]
	for i, node := range path.Nodes {
		var score float32
		for k := range hidden.Data {
			score += node.Weight.Data[k] * hidden.Data[k]
		}

		prob := 1.0 / (1.0 + math.Exp(float64(-score)))

		grad := float32(prob) - path.Directions[i]

		for k := range node.Weight.Data {
			AtomicAddFloat32(&node.Weight.Grad[k], grad*hidden.Data[k])
			AtomicAddFloat32(&hidden.Grad[k], grad*node.Weight.Data[k])
		}
	}
}

func HSLoss(hs *HierarchicalSoftmax, hidden *Tensor, targetID int) *Tensor {
	path := hs.Paths[targetID]
	loss := 0.0
	for i, node := range path.Nodes {
		var score float32
		for k := range hidden.Data {
			score += node.Weight.Data[k] * hidden.Data[k]
		}

		prob := 1.0 / (1.0 + math.Exp(float64(-score)))

		if path.Directions[i] == 1.0 {
			loss -= math.Log(prob + 1e-9)
		} else {
			loss -= math.Log(1.0 - prob + 1e-9)
		}
	}
	res := NewTensor([]float32{float32(loss)}, []int{1}, true)
	res.Op = HSLossOp{HS: hs, Target: targetID}
	res.Parents = []*Tensor{hidden}
	return res
}
