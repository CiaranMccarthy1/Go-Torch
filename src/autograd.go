package gotorch

import "fmt"

type Operation interface {
	Backward(t *Tensor)
}

func (t *Tensor) Backward() {
	if t.gradStorage == nil {
		isScalar := len(t.Shape) == 0 || (len(t.Shape) == 1 && t.Shape[0] == 1)

		if !isScalar {
			panic(fmt.Sprintf(
				"cannot autogenerate gradient for non-scalar tensor with shape %v. "+
					"Call SetGrad() with the appropriate gradient shape before Backward()",
				t.Shape,
			))
		}
		t.SetGrad([]float32{1.0})
	}
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
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.Op != nil {
			node.Op.Backward(node)
		}
	}
}
