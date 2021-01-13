import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node,matmul_op


class TestEngineMul(unittest.TestCase):
    def test_mul_by_const(self):
        x2 = Variable(name="x2")
        y = 5 * x2

        grad_x2, = gradients(y, [x2])

        executor = Executor([y, grad_x2])
        x2_val = 2 * np.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x2_val * 5)
        assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)

    def test_mul_two_vars(self):
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        y = x2 * x3

        grad_x2, grad_x3 = gradients(y, [x2, x3])

        executor = Executor([y, grad_x2, grad_x3])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x2_val * x3_val)
        assert np.array_equal(grad_x2_val, x3_val)
        assert np.array_equal(grad_x3_val, x2_val)

    def test_matmul_two_vars(self):
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        y = matmul_op(x2, x3)

        grad_x2, grad_x3 = gradients(y, [x2, x3])

        executor = Executor([y, grad_x2, grad_x3])
        x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

        expected_yval = np.matmul(x2_val, x3_val)
        expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
        expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_yval)
        assert np.array_equal(grad_x2_val, expected_grad_x2_val)
        assert np.array_equal(grad_x3_val, expected_grad_x3_val)