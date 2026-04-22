package gotorch

type Operation interface {
	Backward(t *Tensor)
}

func (t *Tensor) Backward() {
	if t.gradStorage == nil {
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
