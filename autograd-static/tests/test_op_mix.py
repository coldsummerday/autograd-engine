import  unittest
import numpy as np

from  autograd.engine import Variable,Executor,gradients,Node


class TestEngineMix(unittest.TestCase):
    def test_add_mul_mix_1(self):
        x1 = Variable(name="x1")
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        y = x1 + x2 * x3 * x1

        grad_x1, grad_x2, grad_x3 = gradients(y, [x1, x2, x3])

        executor = Executor([y, grad_x1, grad_x2, grad_x3])
        x1_val = 1 * np.ones(3)
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x1: x1_val, x2: x2_val, x3: x3_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x1_val + x2_val * x3_val)
        assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
        assert np.array_equal(grad_x2_val, x3_val * x1_val)
        assert np.array_equal(grad_x3_val, x2_val * x1_val)

    def test_add_mul_mix_2(self):
        x1 = Variable(name="x1")
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        x4 = Variable(name="x4")
        y = x1 + x2 * x3 * x4

        grad_x1, grad_x2, grad_x3, grad_x4 = gradients(y, [x1, x2, x3, x4])

        executor =Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
        x1_val = 1 * np.ones(3)
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        x4_val = 4 * np.ones(3)
        y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(
            feed_dict={x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val})

        assert isinstance(y, Node)
        assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
        assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
        assert np.array_equal(grad_x2_val, x3_val * x4_val)
        assert np.array_equal(grad_x3_val, x2_val * x4_val)
        assert np.array_equal(grad_x4_val, x2_val * x3_val)

    def test_add_mul_mix_3(self):
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        z = x2 * x2 + x2 + x3 + 3
        y = z * z + x3

        grad_x2, grad_x3 = gradients(y, [x2, x3])

        executor = Executor([y, grad_x2, grad_x3])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

        z_val = x2_val * x2_val + x2_val + x3_val + 3
        expected_yval = z_val * z_val + x3_val
        expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
        expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_yval)
        assert np.array_equal(grad_x2_val, expected_grad_x2_val)
        assert np.array_equal(grad_x3_val, expected_grad_x3_val)

    def test_grad_of_grad(self):
        x2 = Variable(name="x2")
        x3 = Variable(name="x3")
        y = x2 * x2 + x2 * x3

        grad_x2, grad_x3 = gradients(y, [x2, x3])
        grad_x2_x2, grad_x2_x3 = gradients(grad_x2, [x2, x3])

        executor = Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        expected_yval = x2_val * x2_val + x2_val * x3_val
        expected_grad_x2_val = 2 * x2_val + x3_val
        expected_grad_x3_val = x2_val
        expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
        expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

        assert isinstance(y, Node)
        assert np.array_equal(y_val, expected_yval)
        assert np.array_equal(grad_x2_val, expected_grad_x2_val)
        assert np.array_equal(grad_x3_val, expected_grad_x3_val)
        assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
        assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)
