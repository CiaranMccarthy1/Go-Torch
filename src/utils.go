package gotorch

import (
	"math"
	"sync/atomic"
	"unsafe"
)

func AtomicAddFloat32(addr *float32, delta float32) {
	for {
		oldBits := atomic.LoadUint32((*uint32)(unsafe.Pointer(addr)))
		oldVal := math.Float32frombits(oldBits)
		newVal := oldVal + delta
		newBits := math.Float32bits(newVal)
		if atomic.CompareAndSwapUint32((*uint32)(unsafe.Pointer(addr)), oldBits, newBits) {
			return
		}
	}
}

func Transpose(t *Tensor) *Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = t.Data[r*cols+c]
		}
	}
	return NewTensor(out, []int{cols, rows}, t.ReqGrad)
}
