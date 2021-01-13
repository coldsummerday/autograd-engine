import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node


class TestEngineAdd(unittest.TestCase):
    def test_identity(self):
        x2 = Variable(name='x2')
        y = x2

        grad_x2, = gradients(y, [x2])

        executor = Executor([y, grad_x2])
        x2_val = 2 * np.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})
        assert isinstance(y, Node)
        assert np.array_equal(y_val, x2_val)
        assert np.array_equal(grad_x2_val, np.ones_like(x2_val))

    def test_add_by_const(self):
        x2 = Variable("x2")
        y = 5+x2
        grad_x2, = gradients(y,[x2])
        executor = Executor([y,grad_x2])
        x2_val = 2 * np.ones(3)
        y_val,grad_x2_val = executor.run(feed_dict={x2:x2_val})
        assert isinstance(y,Node)
        assert np.array_equal(y_val,x2_val+5)
        assert np.array_equal(grad_x2_val,np.ones_like(x2_val))

    def test_neg(self):
        x1 = Variable(name='x1')
        x2 = Variable(name='x2')

        y = -x2 + x1

        grad_x1, grad_x2 = gradients(y, [x1, x2])
        executor = Executor([y, grad_x1, grad_x2])
        x2_val = 2 * np.ones(3)
        x1_val = 3 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, -x2_val + x1_val)
        assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))
        assert np.array_equal(grad_x1_val, np.ones_like(x1_val))

    def test_add_two_vars(self):
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        y = x2 + x3

        grad_x2, grad_x3 = gradients(y, [x2, x3])

        executor = Executor([y, grad_x2, grad_x3])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x2_val + x3_val)
        assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
        assert np.array_equal(grad_x3_val, np.ones_like(x3_val))



