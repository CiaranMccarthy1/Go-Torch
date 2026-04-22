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
	backend := resolveBackend(t, hidden)
	hiddenData := hidden.Data()
	if len(hiddenData) == 0 {
		return
	}

	scale := float32(1.0)
	gradOut := t.Grad()
	if len(gradOut) == 1 {
		scale = gradOut[0]
	}

	if hidden.ReqGrad && hidden.gradStorage == nil {
		hidden.gradStorage = backend.ZeroStorage(len(hiddenData))
	}

	path := op.HS.Paths[op.Target]
	for i, node := range path.Nodes {
		nodeData := node.Weight.Data()
		var score float32
		for k := range hiddenData {
			score += nodeData[k] * hiddenData[k]
		}

		prob := 1.0 / (1.0 + math.Exp(float64(-score)))

		grad := (float32(prob) - path.Directions[i]) * scale

		if node.Weight.ReqGrad {
			if node.Weight.gradStorage == nil {
				node.Weight.gradStorage = backend.ZeroStorage(len(nodeData))
			}
			nodeGrad := make([]float32, len(nodeData))
			for k := range nodeData {
				nodeGrad[k] = grad * hiddenData[k]
			}
			backend.AddInPlace(node.Weight.gradStorage, backend.CopyToDevice(nodeGrad))
		}

		if hidden.ReqGrad {
			hiddenGrad := make([]float32, len(hiddenData))
			for k := range hiddenData {
				hiddenGrad[k] = grad * nodeData[k]
			}
			backend.AddInPlace(hidden.gradStorage, backend.CopyToDevice(hiddenGrad))
		}
	}
}

func HSLoss(hs *HierarchicalSoftmax, hidden *Tensor, targetID int) *Tensor {
	backend := resolveBackend(hidden)
	hiddenData := hidden.Data()
	path := hs.Paths[targetID]
	loss := 0.0
	for i, node := range path.Nodes {
		nodeData := node.Weight.Data()
		var score float32
		for k := range hiddenData {
			score += nodeData[k] * hiddenData[k]
		}

		prob := 1.0 / (1.0 + math.Exp(float64(-score)))

		if path.Directions[i] == 1.0 {
			loss -= math.Log(prob + 1e-9)
		} else {
			loss -= math.Log(1.0 - prob + 1e-9)
		}
	}
	res := NewTensorWithBackend([]float32{float32(loss)}, []int{1}, true, backend)
	res.Op = HSLossOp{HS: hs, Target: targetID}
	res.Parents = []*Tensor{hidden}
	return res
}
