import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node


class TestEngineDiv(unittest.TestCase):
    def test_div_two_vars(self):
        x1 = Variable(name='x1')
        x2 = Variable(name='x2')

        y = x1 / x2

        grad_x1, grad_x2 = gradients(y, [x1, x2])

        executor = Executor([y, grad_x1, grad_x2])
        x1_val = 2 * np.ones(3)
        x2_val = 5 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x1_val / x2_val)
        assert np.array_equal(grad_x1_val, np.ones_like(x1_val) / x2_val)
        assert np.array_equal(grad_x2_val, -x1_val / (x2_val * x2_val))

    def test_div_by_const(self):
        x2 = Variable(name="x2")
        y = 5 / x2

        grad_x2, = gradients(y, [x2])

        executor = Executor([y, grad_x2])
        x2_val = 2 * np.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, 5 / x2_val)

        assert np.array_equal(grad_x2_val, -5 / (x2_val * x2_val))