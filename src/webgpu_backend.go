package gotorch

import "errors"

const webGPUUnavailableMessage = "webgpu backend requires cgo and native WebGPU bindings; unavailable in this pure-Go build"

type WebGPUBackend struct{}

type webGPUStorage struct{}

func (s *webGPUStorage) Data() []float32 {
	panic(webGPUUnavailableMessage)
}

func NewWebGPUBackend() (*WebGPUBackend, error) {
	return nil, errors.New(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) Name() string {
	return "webgpu"
}

func (b *WebGPUBackend) NewStorage(data []float32, size int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) ZeroStorage(size int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) CopyToHost(s TensorStorage) []float32 {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) CopyToDevice(data []float32) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) Transpose(a TensorStorage, rows, cols int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) MatMulForward(a, c TensorStorage, m, k, n int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) MatMulBackward(a, c, gradOut TensorStorage, m, k, n int, needGradA, needGradB bool) (TensorStorage, TensorStorage) {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) ReLUForward(a TensorStorage) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) ReLUBackward(input, gradOut TensorStorage) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) EmbedForward(weights, indices TensorStorage, batch, dim int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) EmbedBackward(weights, indices, gradOut TensorStorage, batch, dim int) TensorStorage {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) AddInPlace(dst TensorStorage, src TensorStorage) {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) ZeroBuffer(s TensorStorage) {
	panic(webGPUUnavailableMessage)
}

func (b *WebGPUBackend) AdamStep(param, grad, m, v TensorStorage, lr, beta1, beta2, eps float32, step int) {
	panic(webGPUUnavailableMessage)
}

var _ Backend = (*WebGPUBackend)(nil)
