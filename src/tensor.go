package gotorch

type Tensor struct {
	Data    []float32
	Shape   []int
	Grad    []float32
	Parents []*Tensor
	Op      Operation
	ReqGrad bool
	Backend Backend
	storage TensorStorage
}

func NewTensor(data []float32, shape []int, reqGrad bool) *Tensor {
	return NewTensorWithBackend(data, shape, reqGrad, DefaultBackend())
}

func NewTensorWithBackend(data []float32, shape []int, reqGrad bool, backend Backend) *Tensor {
	if backend == nil {
		backend = DefaultBackend()
	}

	size := 1
	for _, dim := range shape {
		size *= dim
	}

	storage := backend.NewStorage(data, size)
	tensorData := storage.Data()

	t := &Tensor{Data: tensorData, Shape: shape, ReqGrad: reqGrad, Backend: backend, storage: storage}
	if reqGrad {
		t.Grad = make([]float32, size)
	}
	return t
}

func (t *Tensor) values() []float32 {
	if t.Backend == nil {
		t.Backend = DefaultBackend()
	}

	if t.storage == nil {
		t.storage = t.Backend.NewStorage(t.Data, len(t.Data))
	} else {
		stored := t.storage.Data()
		if len(stored) != len(t.Data) {
			t.storage = t.Backend.NewStorage(t.Data, len(t.Data))
		} else if len(t.Data) > 0 && len(stored) > 0 && &stored[0] != &t.Data[0] {
			t.storage = t.Backend.NewStorage(t.Data, len(t.Data))
		}
	}

	t.Data = t.storage.Data()
	return t.Data
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	}
}
