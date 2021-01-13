import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node


class TestEngineSub(unittest.TestCase):
    def test_sub_by_const(self):
        x2 = Variable(name='x2')
        y = 3 - x2
        grad_x2, = gradients(y, [x2])
        executor = Executor([y, grad_x2])
        x2_val = 2 * np.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, 3 - x2_val)
        assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))
