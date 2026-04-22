package gotorch

type Tensor struct {
	storage     TensorStorage
	gradStorage TensorStorage
	Shape       []int
	Parents     []*Tensor
	Op          Operation
	ReqGrad     bool
	Backend     Backend
}

func NewTensor(data []float32, shape []int, reqGrad bool) *Tensor {
	return NewTensorWithBackend(data, shape, reqGrad, DefaultBackend())
}

func NewTensorWithBackend(data []float32, shape []int, reqGrad bool, backend Backend) *Tensor {
	if backend == nil {
		backend = DefaultBackend()
	}

	size := shapeSize(shape)
	storage := backend.NewStorage(data, size)
	return &Tensor{storage: storage, Shape: append([]int(nil), shape...), ReqGrad: reqGrad, Backend: backend}
}

func newTensorFromStorage(storage TensorStorage, shape []int, reqGrad bool, backend Backend) *Tensor {
	if backend == nil {
		backend = DefaultBackend()
	}
	return &Tensor{storage: storage, Shape: append([]int(nil), shape...), ReqGrad: reqGrad, Backend: backend}
}

func (t *Tensor) ensureBackend() Backend {
	if t.Backend == nil {
		t.Backend = DefaultBackend()
	}
	return t.Backend
}

func (t *Tensor) ensureStorage() TensorStorage {
	if t.storage == nil {
		t.storage = t.ensureBackend().ZeroStorage(shapeSize(t.Shape))
	}
	return t.storage
}

func (t *Tensor) Data() []float32 {
	if t == nil {
		return nil
	}
	if t.storage == nil {
		return nil
	}
	return t.ensureBackend().CopyToHost(t.storage)
}

func (t *Tensor) Grad() []float32 {
	if t == nil || t.gradStorage == nil {
		return nil
	}
	return t.ensureBackend().CopyToHost(t.gradStorage)
}

func (t *Tensor) SetGrad(data []float32) {
	t.gradStorage = t.ensureBackend().CopyToDevice(data)
}

func (t *Tensor) ensureGradStorage(size int) TensorStorage {
	if t.gradStorage == nil {
		t.gradStorage = t.ensureBackend().ZeroStorage(size)
	}
	return t.gradStorage
}

func (t *Tensor) ZeroGrad() {
	if t.gradStorage != nil {
		t.ensureBackend().ZeroBuffer(t.gradStorage)
	}
}

func shapeSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}
