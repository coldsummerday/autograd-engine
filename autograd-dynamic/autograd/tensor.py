#!/usr/bin/python3
import  numpy as np

from typing import List,NamedTuple,Callable,Union

class Dependency(NamedTuple):
    tensor:'Tensor'
    grad_fn:Callable[[np.ndarray],np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable:Arrayable)->np.ndarray:
    if isinstance(arrayable,np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor',float,np.ndarray]

def ensure_tensor(tensorable:Tensorable)->'Tensor':
    if isinstance(tensorable,Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Tensor():
    def __init__(self,
                 data:np.ndarray,
                 requires_grad:bool=False,
                 depends_on: List[Dependency] =None)->None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape

        self.grad = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self)->np.ndarray:
        return self._data

    @data.setter
    def data(self,new_data:np.ndarray)->None:
        self._data = new_data
        self.grad = None

    def __repr__(self)->str:
        return f"Tensor({self._data}),requires_grad={self.requires_grad}"

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self._data))

    def __add__(self, other)->'Tensor':
        return add(self,ensure_tensor(other))

    def __radd__(self, other)->'Tensor':
        return add(ensure_tensor(other),self)

    def __iadd__(self, other)->'Tensor':
        """when we do t += other"""
        self.data = self.data +ensure_tensor(other).data
        return self

    def __isub__(self, other)->'Tensor':
        """when we do t-=other"""
        self.data = self.data -ensure_tensor(other).data
        return self

    def __imul__(self, other)->'Tensor':
        """when we do t *=other"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __mul__(self, other)->'Tensor':
        return mul(self,ensure_tensor(other))

    def __rmul__(self, other)->'Tensor':
        return mul(ensure_tensor(other),self)

    def __matmul__(self, other)->'Tensor':
        return _matmul(self,other)

    def __sub__(self, other)->'Tensor':
        return sub(self,ensure_tensor(other))

    def __rsub__(self, other)->'Tensor':
        return sub(ensure_tensor(other),self)



    def __neg__(self)->'Tensor':
        return neg(self)

    def __getitem__(self, idxs)->'Tensor':
        return _slice(self,idxs)





    def backward(self,grad:'Tensor'=None)->None:
        assert self.requires_grad,"called backward on non-requires-grad tensor"

        if grad==None:
            if self.shape==():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be speified for non-0-tensor")
        ##梯度相加
        self.grad.data = self.grad.data + grad.data

        #计算图中的依赖也进行计算
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self):
        return tensor_sum(self)








def tensor_sum(t:Tensor)->Tensor:
    """
        Takes a tensor and returns the 0-tensor
        that's the sum of all its elements.

    返回所有元素之和
    """
    data = t.data.sum()
    requires_grad = t.requires_grad
    if requires_grad:
        ##将梯度传递回去
        def grad_fn(grad:np.ndarray)->np.ndarray:
            """
                        grad is necessarily a 0-tensor, so each input element
                        contributes that much
            """
            return grad * np.ones_like(t.data)
        depends_on = [Dependency(t,grad_fn)]
    else:
        depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def add(t1:Tensor,t2:Tensor)->Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on:List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad:np.ndarray)->np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim

            ##将后面的dim合并
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i,dim in enumerate(t1.shape):
                if dim ==1 :
                    grad = grad.sum(axis=i,keepdims=True)
            return grad
        depends_on.append(Dependency(t1,grad_fn=grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            #(1,5)这种情况，
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def mul(t1:Tensor,t2:Tensor)->Tensor:
    data = t1.data * t2.data

    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on:List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad:np.ndarray)->np.ndarray:
            grad = grad * t2.data


            ##borad_cast 问题
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1,grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad:np.ndarray)->np.ndarray:
            grad = grad * t1.data

            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t2,grad_fn2))
    return Tensor(
        data,requires_grad,depends_on
    )


def neg(t:Tensor)->Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t,lambda x:-x)]
    else:
        depends_on = []
    return Tensor(data,requires_grad,depends_on)

def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return add(t1, neg(t2))


def _matmul(t1:Tensor,t2:Tensor)->Tensor:
    """

    :param t1:
    :param t2:
     if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3

    :return:
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on:List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad:np.ndarray)->np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1,grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad:np.ndarray)->np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2,grad_fn2))
    return Tensor(data,requires_grad,depends_on)


def _slice(t:Tensor,idx)->Tensor:
    data = t.data[idx]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad:np.ndarray)->np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idx]=grad
            return bigger_grad
        depends_on = Dependency(t,grad_fn)
    else:
        depends_on = []
    return Tensor(data,requires_grad,depends_on)

