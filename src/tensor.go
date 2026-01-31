package gotorch

type Tensor struct {
	Data    []float32
	Shape   []int
	Grad    []float32
	Parents []*Tensor
	Op      Operation
	ReqGrad bool
}

func NewTensor(data []float32, shape []int, reqGrad bool) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if data == nil {
		data = make([]float32, size)
	}
	t := &Tensor{Data: data, Shape: shape, ReqGrad: reqGrad}
	if reqGrad {
		t.Grad = make([]float32, size)
	}
	return t
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	}
}
